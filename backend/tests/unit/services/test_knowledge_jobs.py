from services.knowledge_jobs import KnowledgeJobStore


def test_create_returns_queued_job():
    store = KnowledgeJobStore()
    job = store.create("report.pdf", "mining")

    assert job.stage == "queued"
    assert job.filename == "report.pdf"
    assert job.domain == "mining"


def test_get_returns_dict_snapshot():
    store = KnowledgeJobStore()
    job = store.create("report.pdf", "mining")

    snapshot = store.get(job.job_id)

    assert snapshot["job_id"] == job.job_id
    assert snapshot["stage"] == "queued"


def test_get_unknown_job_returns_none():
    store = KnowledgeJobStore()
    assert store.get("does-not-exist") is None


def test_update_mutates_existing_job():
    store = KnowledgeJobStore()
    job = store.create("report.pdf", "mining")

    store.update(job.job_id, stage="embedding", chunks_total=10, chunks_embedded=3)

    snapshot = store.get(job.job_id)
    assert snapshot["stage"] == "embedding"
    assert snapshot["chunks_total"] == 10
    assert snapshot["chunks_embedded"] == 3


def test_update_unknown_job_is_a_noop():
    store = KnowledgeJobStore()
    store.update("does-not-exist", stage="done")  # should not raise
