import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from config import settings
from retrieval import Retriever
from retrieval.chunker import Chunker
from retrieval.converter import convert_to_markdown
from services.knowledge_jobs import knowledge_job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB

# Conversion and chunking run fully concurrently across files (local CPU work).
# Only the embedding + store step is serialized, since that's the part that
# calls the rate-limited Mistral API.
_embedding_lock = asyncio.Lock()


@router.get("")
async def list_knowledge():
    """List all unique source documents in the industry_knowledge collection."""
    retriever = Retriever()
    sources = retriever.list_sources(settings.stores.chroma_knowledge_collection)
    return {"documents": sources, "total": len(sources)}


async def _process_job(job_id: str, filename: str, raw: bytes, domain: str, created_at: float) -> None:
    """Background pipeline: convert -> chunk -> embed -> store, updating job stage as it goes."""
    try:
        logger.info("Job %s: converting '%s' (%d bytes)", job_id, filename, len(raw))
        knowledge_job_store.update(job_id, stage="converting")
        markdown = await asyncio.to_thread(convert_to_markdown, filename, raw)

        logger.info("Job %s: chunking %d chars of markdown", job_id, len(markdown))
        knowledge_job_store.update(job_id, stage="chunking")
        chunker = Chunker(settings.retrieval.chunk_size, settings.retrieval.chunk_overlap)
        docs = chunker.chunk_document(markdown, source=filename, domain=domain)
        if not docs:
            raise ValueError("Document produced no chunks after splitting.")

        if knowledge_job_store.deleted_after(filename, created_at):
            logger.info("Job %s: '%s' was deleted while queued, skipping store", job_id, filename)
            knowledge_job_store.update(job_id, stage="cancelled")
            return

        logger.info("Job %s: embedding %d chunks", job_id, len(docs))
        knowledge_job_store.update(job_id, stage="embedding", chunks_total=len(docs))

        def on_progress(done: int, total: int) -> None:
            logger.info("Job %s: embedded %d/%d chunks", job_id, done, total)
            knowledge_job_store.update(job_id, chunks_embedded=done)

        retriever = Retriever()
        async with _embedding_lock:
            # Re-check right before writing: the file may have been deleted
            # while this job was waiting on the embedding lock.
            if knowledge_job_store.deleted_after(filename, created_at):
                logger.info("Job %s: '%s' was deleted while queued, skipping store", job_id, filename)
                knowledge_job_store.update(job_id, stage="cancelled")
                return
            await asyncio.to_thread(
                retriever.add,
                settings.stores.chroma_knowledge_collection,
                docs,
                on_progress,
            )

        logger.info("Job %s: stored %d chunks for '%s'", job_id, len(docs), filename)
        knowledge_job_store.update(job_id, stage="done", chunks_added=len(docs))
    except Exception as exc:
        logger.exception("Job %s: failed converting/embedding '%s'", job_id, filename)
        knowledge_job_store.update(job_id, stage="error", error=str(exc))


@router.post("")
async def upload_knowledge(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    domain: str = Form(...),
):
    """Create a background job to convert, chunk, embed, and store a document.

    Accepts .md, .txt, .pdf, .docx, .pptx. Returns immediately with a job_id;
    poll GET /knowledge/jobs/{job_id} for progress. Conversion and chunking run
    concurrently across files; only the embedding/store step is serialized to
    keep load on the Mistral API predictable for multi-file uploads.
    """
    raw = await file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 50MB upload limit.")

    job = knowledge_job_store.create(file.filename, domain)
    background_tasks.add_task(
        _process_job, job.job_id, file.filename, raw, domain, job.created_at
    )
    return {"job_id": job.job_id}


@router.get("/jobs/{job_id}")
async def get_knowledge_job(job_id: str):
    """Poll the status of an in-flight or completed knowledge upload job."""
    job = knowledge_job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/{source}")
async def delete_knowledge(source: str):
    """Delete all chunks for a named source from the industry_knowledge collection.

    Also tombstones *source* so any upload job for it that was queued before
    this delete (and hasn't reached the embed/store step yet) is suppressed
    instead of re-inserting the chunks the user just removed.
    """
    retriever = Retriever()
    deleted = retriever.delete_by_source(settings.stores.chroma_knowledge_collection, source)
    knowledge_job_store.record_delete(source)
    return {"source": source, "chunks_deleted": deleted}
