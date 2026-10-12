"""Runtime settings (environment variables; see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env = _BACKEND.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    nlp_root: Path = field(default_factory=lambda: Path(os.environ.get("NLP_ROOT", _BACKEND.parents[1])))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if o.strip()))
    preload_model: bool = field(default_factory=lambda: os.environ.get("PRELOAD_MODEL", "true").lower() == "true")
    max_text_chars: int = field(default_factory=lambda: int(os.environ.get("MAX_TEXT_CHARS", "2000")))
    max_batch_rows: int = field(default_factory=lambda: int(os.environ.get("MAX_BATCH_ROWS", "2000")))
    max_upload_mb: float = field(default_factory=lambda: float(os.environ.get("MAX_UPLOAD_MB", "5")))

    @property
    def telugu_outputs(self) -> Path:
        return self.nlp_root / "outputs"

    @property
    def multi(self) -> Path:
        return self.nlp_root / "sentiment_app" / "artifacts" / "multilingual"


settings = Settings()
