"""In-memory job tracking for industry-knowledge uploads.

Each uploaded file becomes a short-lived job so the frontend can poll
its stage (queued -> converting -> chunking -> embedding -> done/error)
instead of blocking on one long request. Jobs are bookkeeping only —
not worth a SQLite schema since they don't need to survive a restart.
"""
import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Literal

JobStage = Literal["queued", "converting", "chunking", "embedding", "done", "error"]


@dataclass
class KnowledgeJob:
    job_id: str
    filename: str
    domain: str
    stage: JobStage = "queued"
    chunks_total: int | None = None
    chunks_embedded: int | None = None
    chunks_added: int | None = None
    error: str | None = None


class KnowledgeJobStore:
    """Thread-safe in-memory job registry.

    Updates happen from a worker thread (via asyncio.to_thread); reads
    happen from the event loop — both go through the same lock.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, KnowledgeJob] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, domain: str) -> KnowledgeJob:
        job = KnowledgeJob(job_id=str(uuid.uuid4()), filename=filename, domain=domain)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return asdict(job) if job else None

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)


# Process-wide singleton — jobs need to be visible across requests.
knowledge_job_store = KnowledgeJobStore()
