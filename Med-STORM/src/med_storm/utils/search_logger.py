"""search_logger.py
--------------------
Utility responsible for persisting every external search performed by Med-STORM
in a YAML file for full reproducibility and audit-trail (PRISMA identification stage).

Files are stored under ``search_logs/<YYYY-MM-DD>/<run_id>/`` so that
multiple engines executed the same day do not collide. A *run_id* is derived
from :pyfunc:`uuid.uuid4` at first import and can be overridden via the
``MEDSTORM_RUN_ID`` environment variable (useful in tests).

Each connector call appends a small YAML fragment containing:

* connector: ``pubmed``, ``serper`` …
* query: raw string sent
* filters: dict of filters/params
* timestamp: ISO–8601
* results: integer number of hits returned
"""
from __future__ import annotations

import os
import uuid
try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – fallback to JSON lines
    yaml = None  # type: ignore
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class SearchLogger:  # pragma: no cover – utility, tested indirectly
    """Persist search metadata to the *search_logs* folder."""

    # Run-wide identifier (exported as env var in reports to correlate artefacts)
    RUN_ID: str = os.getenv("MEDSTORM_RUN_ID", uuid.uuid4().hex[:8])

    BASE_DIR: Path = Path(os.getenv("SEARCH_LOG_DIR", "search_logs"))

    @classmethod
    def _get_run_dir(cls) -> Path:
        date_dir = cls.BASE_DIR / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        run_dir = date_dir / cls.RUN_ID
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    @classmethod
    def log(
        cls,
        *,
        connector: str,
        query: str,
        filters: Dict[str, Any] | None = None,
        results: int | None = None,
        raw_response: Any | None = None,
    ) -> None:
        """Append a YAML entry with search metadata.

        The file is named ``<connector>.yaml`` and each search request is
        appended as a new document (``---`` separator) so that multiple calls
        are preserved in chronological order.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "connector": connector,
            "query": query,
            "filters": filters or {},
            "results": results,
        }

        run_dir = cls._get_run_dir()
        file_path = run_dir / f"{connector}.yaml"

        # Append document (YAML if available, else JSON)
        with file_path.open("a", encoding="utf-8") as fh:
            if yaml is not None:
                yaml.safe_dump(entry, fh, allow_unicode=True, sort_keys=False)
                fh.write("---\n")
            else:
                import json
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Optionally persist full raw payload for complete reproducibility
        if raw_response is not None and os.getenv("MEDSTORM_SAVE_RAW_SEARCH_JSON", "0") == "1":
            import json, gzip, time
            ts = int(time.time())
            json_path = run_dir / f"{connector}_{ts}.json.gz"
            with gzip.open(json_path, "wt", encoding="utf-8") as gz:
                json.dump(raw_response, gz, ensure_ascii=False, indent=2, default=str) 