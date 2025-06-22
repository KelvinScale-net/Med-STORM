"""conflict_scraper.py
---------------------------------
Utility to extract *conflict of interest* and *funding* statements from
primary literature identifiers (currently PubMed).

This data feeds the Evidence-synthesis quality-gate rows **conflicts/funding**
and is used downstream by the Risk-of-Bias and GRADE modules.

Design goals
~~~~~~~~~~~~
1. Minimal external dependencies – relies on *aiohttp* already used elsewhere.
2. Batched & cached requests – avoids exceeding NCBI E-utils limits.
3. Deterministic parsing – no fuzzy NLP; we inspect explicit XML tags.

Public API
~~~~~~~~~~
>>> from med_storm.ingestion.conflict_scraper import ConflictFundingScraper
>>> results = await ConflictFundingScraper().scrape_pmids(["12345", "67890"])

Return schema per PMID::
    {
        "pmid": "12345",
        "conflict_statement": "Authors declare no competing interests.",
        "funding_statement": "Supported by NIH grant…",
        "flags": {
            "industry_sponsored": False,
            "no_disclosure": False
        }
    }
"""
from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from med_storm.utils.cache import ultra_cache
from med_storm.utils.search_logger import SearchLogger

logger = logging.getLogger(__name__)


class ConflictFundingScraper:
    """Extract *Conflicts of interest* and *Funding* from PubMed records."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        self.email = email
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    # ------------------------------------------------------------------
    # Public orchestrator – exposed to engines
    # ------------------------------------------------------------------
    async def scrape_pmids(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Return conflict/funding information for each PMID in *pmids*."""
        if not pmids:
            return []

        logger.info("Scraping conflict/funding statements for %s PMIDs", len(pmids))

        chunks: List[List[str]] = self._chunk(pmids, 200)
        tasks = [self._fetch_and_parse_chunk(chunk) for chunk in chunks]
        results_nested = await asyncio.gather(*tasks, return_exceptions=False)
        # Flatten list of lists
        combined: List[Dict[str, Any]] = [item for sub in results_nested for item in sub]
        return combined

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk(seq: List[str], size: int) -> List[List[str]]:
        return [seq[i : i + size] for i in range(0, len(seq), size)]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    @ultra_cache(expiry_seconds=86400 * 7)  # 1-week cache
    async def _fetch_and_parse_chunk(self, chunk: List[str]) -> List[Dict[str, Any]]:
        """Fetch XML for a chunk of PMIDs and parse into dicts."""
        session = await self._get_session()
        params = {
            "db": "pubmed",
            "id": ",".join(chunk),
            "retmode": "xml",
            "retmax": len(chunk),
        }
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email

        SearchLogger.log(connector="pubmed_conflict", query="pmid_fetch", filters=params, results=len(chunk))

        async with session.post(f"{self.BASE_URL}/efetch.fcgi", params=params) as resp:
            resp.raise_for_status()
            xml_text = await resp.text()

        root = ET.fromstring(xml_text)
        records: List[Dict[str, Any]] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID") or "unknown"
            conflict_stmt = self._extract_conflict(article)
            funding_stmt = self._extract_funding(article)
            flags = self._flag_risk(conflict_stmt, funding_stmt)
            records.append({
                "pmid": pmid,
                "conflict_statement": conflict_stmt,
                "funding_statement": funding_stmt,
                "flags": flags,
                "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        return records

    # ------------------------------------------------------------------
    # Parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_conflict(article_node: ET.Element) -> str:
        # Common patterns
        # 1) <AbstractText Label="Conflict of interest">...
        text = article_node.findtext(".//Abstract/AbstractText[@Label='Conflict of interest']")
        if text:
            return text.strip()
        # 2) <PublicationTypeList><PublicationType>Research Support, Non-U.S. Gov't</PublicationType>
        #    Not ideal, fallback to empty.
        return ""

    @staticmethod
    def _extract_funding(article_node: ET.Element) -> str:
        # Extract from GrantList
        grants = []
        for grant in article_node.findall(".//GrantList/Grant"):
            agency = grant.findtext("Agency") or ""
            grant_id = grant.findtext("GrantID") or ""
            grants.append(f"{agency} {grant_id}".strip())
        if grants:
            return "; ".join(grants)
        return ""

    @staticmethod
    def _flag_risk(conflict_stmt: str, funding_stmt: str) -> Dict[str, bool]:
        """Generate heuristic risk flags."""
        stmt_combined = f"{conflict_stmt} {funding_stmt}".lower()

        no_disclosure_patterns = [
            r"no disclosure",  # explicit
            r"not disclosed",
            r"undisclosed",
        ]

        no_disclosure = stmt_combined.strip() == "" or any(re.search(pat, stmt_combined) for pat in no_disclosure_patterns)

        return {
            "industry_sponsored": any(word in stmt_combined for word in ["pfizer", "novartis", "merck", "lilly", "astellas", "gsk", "sanofi"]),
            "no_disclosure": no_disclosure,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close() 