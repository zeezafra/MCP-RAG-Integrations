"""Replaceable adapters for heterogeneous knowledge sources."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


class KnowledgeAdapter(Protocol):
    """Small contract shared by every retrieval source."""

    def read(self) -> str:
        """Return the source's current records as newline-delimited text."""


@dataclass(frozen=True)
class TextDocumentAdapter:
    """Read a UTF-8 document from disk."""

    path: Path

    def read(self) -> str:
        if not self.path.is_file():
            raise FileNotFoundError(f"Knowledge document not found: {self.path}")
        return self.path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class SQLiteFaqAdapter:
    """Query FAQ records through SQLite instead of a document reader."""

    records: tuple[tuple[str, str], ...]

    def read(self) -> str:
        with sqlite3.connect(":memory:") as database:
            database.execute(
                "CREATE TABLE faq (question TEXT NOT NULL, answer TEXT NOT NULL)"
            )
            database.executemany(
                "INSERT INTO faq (question, answer) VALUES (?, ?)",
                self.records,
            )
            rows = database.execute(
                "SELECT question, answer FROM faq ORDER BY rowid"
            ).fetchall()
        return "\n".join(
            f"Q: {question} A: {answer}" for question, answer in rows
        )


@dataclass(frozen=True)
class LiveStatusAdapter:
    """Generate a current operations feed from runtime configuration."""

    def read(self) -> str:
        checked_at = datetime.now(UTC).isoformat(timespec="seconds")
        response_days = os.getenv("SUPPORT_RESPONSE_DAYS", "1")
        inspection_days = os.getenv("RETURN_INSPECTION_DAYS", "2")
        disruption = os.getenv("SHIPPING_SERVICE_DISRUPTION", "none").strip()
        shipping_line = (
            "No shipping service disruptions are currently reported."
            if disruption.lower() in {"", "none", "false", "no"}
            else f"Current shipping disruption: {disruption}"
        )
        return "\n".join(
            [
                f"Status last checked at {checked_at}.",
                (
                    "Support response time is currently within "
                    f"{response_days} business day(s)."
                ),
                (
                    "Return inspections are currently completed within "
                    f"{inspection_days} business day(s) of warehouse receipt."
                ),
                shipping_line,
            ]
        )


FAQ_RECORDS = (
    (
        "How do I report a damaged product?",
        "Contact support within 30 days of delivery and attach clear photos.",
    ),
    (
        "What happens after I report damage?",
        "Support verifies the order and evidence, then sends a return authorization.",
    ),
    (
        "How long do approved refunds take?",
        "Refunds normally reach the original payment method within 5 to 7 business days.",
    ),
    (
        "Do I need the original packaging?",
        "Original packaging is recommended but is not always required for a damaged-item claim.",
    ),
)


def build_adapters(data_dir: Path) -> dict[str, KnowledgeAdapter]:
    """Compose the default document, database, and live-feed adapters."""
    return {
        "products": TextDocumentAdapter(data_dir / "products.txt"),
        "support": SQLiteFaqAdapter(FAQ_RECORDS),
        "policies": TextDocumentAdapter(data_dir / "policies.txt"),
        "status": LiveStatusAdapter(),
    }
