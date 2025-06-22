#!/usr/bin/env python3
"""
Full professional Med-STORM report generator
===========================================

Generates a journal-quality evidence synthesis (PRISMA 2020 compliant) plus
executive summary and saves both JSON and Markdown artefacts.

Example:
    python examples/run_full_report.py "cardiovascular risk management in type 2 diabetes" \
        --patient data/patient_profile.json --output-dir output

Requires API keys in .env for OpenRouter, PubMed, Serper, etc.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import dataclasses
from pydantic import BaseModel  # local import to avoid mandatory dep if not present
from enum import Enum  # local import

# Workspace root → add src to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Optional dependency: python-slugify
# If unavailable, fall back to a minimal slug implementation.
# ---------------------------------------------------------------------------

try:
    from slugify import slugify  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – fallback for missing dep
    import re
    import unicodedata

    def slugify(value: str, allow_unicode: bool = False) -> str:  # type: ignore
        """Fallback slugify implementation (alnum + hyphens)."""
        value = str(value)
        if allow_unicode:
            value = unicodedata.normalize("NFKC", value)
        else:
            value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")

        value = re.sub(r"[^\w\s-]", "", value.lower())
        value = re.sub(r"[\s_-]+", "-", value).strip("-")
        return value[:80]

from med_storm.config import settings
from med_storm.llm.openrouter import OpenRouterLLM
from med_storm.connectors.pubmed import PubMedConnector
from med_storm.connectors.serper import SerperConnector
from med_storm.connectors.local_corpus import UltraLocalCorpusConnector
from med_storm.core.storm_enhanced_engine import StormEnhancedMedicalEngine


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a full Med-STORM report")
    parser.add_argument("topic", type=str, help="Clinical topic / research question")
    parser.add_argument("--patient", type=str, help="Path to JSON file with patient profile")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save artefacts")
    parser.add_argument("--max-personas", type=int, default=6, help="Number of expert personas to generate")
    parser.add_argument("--max-q-per-persona", type=int, default=8, help="Questions per persona")
    return parser.parse_args()


def _build_connectors() -> Dict[str, Any]:
    """Instantiate default knowledge connectors."""
    pubmed_key = getattr(settings, "PUBMED_API_KEY", None) or getattr(settings, "pubmed_api_key", None)
    serper_key = getattr(settings, "SERPER_API_KEY", None) or getattr(settings, "serper_api_key", None)
    qdrant_key = getattr(settings, "QDRANT_API_KEY", None)

    collection_name = getattr(settings, "QDRANT_COLLECTION_NAME", None) or "medstorm_default"

    return {
        "pubmed": PubMedConnector(email=settings.PUBMED_EMAIL, api_key=pubmed_key),
        "serper": SerperConnector(api_key=serper_key),
        "localcorpus": UltraLocalCorpusConnector(
            collection_name=collection_name,
            qdrant_url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
        ),
    }


async def _generate_report(args: argparse.Namespace):
    # Optional patient profile
    patient_profile: Optional[Dict[str, Any]] = None
    if args.patient:
        patient_path = Path(args.patient)
        if not patient_path.exists():
            raise FileNotFoundError(f"Patient profile not found: {patient_path}")
        patient_profile = json.loads(patient_path.read_text(encoding="utf-8"))

    # Instantiate LLM (Gemini-2.5 Flash)
    llm = OpenRouterLLM()

    # Connectors
    connectors = _build_connectors()

    # Engine
    engine = StormEnhancedMedicalEngine(llm_provider=llm, connectors=connectors)

    # Ensure output directory exists
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("🚀 Generating report for: %s", args.topic)
    report = await engine.run(
        topic=args.topic,
        patient_profile=patient_profile,
        enable_systematic_review=True,
        enable_advanced_statistics=True,
        enable_personalized_medicine=patient_profile is not None,
    )

    # Gracefully close connector sessions
    for conn in engine.connectors.values():
        if hasattr(conn, "close_session"):
            try:
                await conn.close_session()
            except Exception as close_err:
                logging.warning("Failed to close connector session: %s", close_err)

    slug = slugify(args.topic)[:80]
    json_file = out_dir / f"{slug}.json"
    md_file = out_dir / f"{slug}.md"

    # Convert dataclass instances to plain dicts for JSON serialization
    def _serialize(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
        if isinstance(obj, list):
            return [_serialize(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, BaseModel):
            return {k: _serialize(v) for k, v in obj.model_dump().items()}
        if isinstance(obj, Enum):
            return obj.value if hasattr(obj, "value") else str(obj)
        return obj

    serializable_report = _serialize(report)
    json_file.write_text(json.dumps(serializable_report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save Markdown
    md_content = (
        f"# {args.topic}\n\n"
        f"## Executive summary\n\n{report['executive_summary']}\n\n"
        f"---\n\n{report['synthesis']['content']}\n"
    )
    md_file.write_text(md_content, encoding="utf-8")

    logging.info("✅ Report saved to: %s and %s", md_file, json_file)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()
    asyncio.run(_generate_report(args))


if __name__ == "__main__":
    main() 