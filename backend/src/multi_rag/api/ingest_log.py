# In-memory ingestion job tracking.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
import uuid


@dataclass
class IngestJob:
    job_id: str
    source_type: str
    title: str
    origin: str
    status: str
    error: str | None
    created_at: datetime


class InMemoryIngestLog:
    def __init__(self) -> None:
        self._jobs: list[IngestJob] = []

    def record_success(self, source_type: str, title: str, origin: str) -> IngestJob:
        job = IngestJob(
            job_id=uuid.uuid4().hex,
            source_type=source_type,
            title=title,
            origin=origin,
            status="completed",
            error=None,
            created_at=datetime.now(UTC),
        )
        self._jobs.append(job)
        return job

    def record_failure(
        self, source_type: str, title: str, origin: str, error: str
    ) -> IngestJob:
        job = IngestJob(
            job_id=uuid.uuid4().hex,
            source_type=source_type,
            title=title,
            origin=origin,
            status="failed",
            error=error,
            created_at=datetime.now(UTC),
        )
        self._jobs.append(job)
        return job

    def list_jobs(self, limit: int = 50) -> list[IngestJob]:
        jobs = sorted(self._jobs, key=lambda item: item.created_at, reverse=True)
        return jobs[:limit]
