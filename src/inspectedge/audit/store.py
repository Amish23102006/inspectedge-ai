"""SQLite audit store for inspections, human reviews and audit events (MVP)."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from inspectedge.schemas import AuditEvent, HumanReview, InspectionResponse

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    decision      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    payload       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id TEXT NOT NULL REFERENCES inspections(inspection_id),
    reviewed_at   TEXT NOT NULL,
    payload       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id  TEXT NOT NULL,
    timestamp  TEXT NOT NULL,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_object ON audit_events(object_id);
"""


class AuditStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as c, c:
            c.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save_inspection(self, r: InspectionResponse) -> None:
        with closing(self._connect()) as c, c:
            c.execute(
                "INSERT INTO inspections VALUES (?, ?, ?, ?, ?)",
                (
                    r.inspection_id,
                    r.created_at.isoformat(),
                    r.decision.value,
                    r.model_version,
                    r.model_dump_json(),
                ),
            )

    def get_inspection(self, inspection_id: str) -> InspectionResponse | None:
        with closing(self._connect()) as c:
            row = c.execute(
                "SELECT payload FROM inspections WHERE inspection_id = ?", (inspection_id,)
            ).fetchone()
        return InspectionResponse.model_validate_json(row[0]) if row else None

    def save_review(self, review: HumanReview) -> None:
        with closing(self._connect()) as c, c:
            c.execute(
                "INSERT INTO reviews (inspection_id, reviewed_at, payload) VALUES (?, ?, ?)",
                (review.inspection_id, review.reviewed_at.isoformat(), review.model_dump_json()),
            )

    def list_reviews(self, inspection_id: str) -> list[HumanReview]:
        with closing(self._connect()) as c:
            rows = c.execute(
                "SELECT payload FROM reviews WHERE inspection_id = ? ORDER BY review_id",
                (inspection_id,),
            ).fetchall()
        return [HumanReview.model_validate_json(r[0]) for r in rows]

    def log_event(self, e: AuditEvent) -> None:
        with closing(self._connect()) as c, c:
            c.execute(
                "INSERT INTO audit_events (object_id, timestamp, payload) VALUES (?, ?, ?)",
                (e.object_id, e.timestamp.isoformat(), e.model_dump_json()),
            )

    def list_events(self, object_id: str) -> list[AuditEvent]:
        with closing(self._connect()) as c:
            rows = c.execute(
                "SELECT payload FROM audit_events WHERE object_id = ? ORDER BY event_id",
                (object_id,),
            ).fetchall()
        return [AuditEvent.model_validate_json(r[0]) for r in rows]
