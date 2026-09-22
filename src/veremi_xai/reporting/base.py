from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


class ReporterError(RuntimeError):
    """Raised by a backend when no report could be produced (no key, network error, quota, bad response)."""


@dataclass
class Report:
    text: str
    backend: str                      # "template" or profile name (groq, gemini, ollama, ...)
    model: str
    prompt_version: str
    evidence_hash: str
    validation: dict = field(default_factory=dict)
    fallback_reason: str | None = None   # set when an LLM was requested but the template had to be used
    from_cache: bool = False
    attempts: int = 0                      # LLM calls made for this report (0 = template only, 2 = corrective retry was needed)
    rejected_llm_text: str | None = None   # LLM output that failed validation (kept for transparency)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        return asdict(self)
