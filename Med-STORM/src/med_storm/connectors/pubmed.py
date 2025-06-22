import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from med_storm.connectors.base import KnowledgeConnector
from med_storm.models.evidence import EvidenceCorpus, EvidenceSource
from med_storm.utils.cache import ultra_cache

# Configure logging
logger = logging.getLogger(__name__)

class PubMedConnector(KnowledgeConnector):
    """
    A connector to interact with the PubMed API, with manual caching logic.
    """
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initializes the PubMedConnector.
        
        Args:
            email (Optional[str]): Your email address for NCBI API requests.
            api_key (Optional[str]): Your NCBI API key.
        """
        self.email = email
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info("PubMedConnector initialized.")
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Initializes and returns an aiohttp.ClientSession."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
        
    async def close_session(self):
        """Closes the aiohttp.ClientSession."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("PubMedConnector session closed.")

    def _add_auth(self, params: Dict[str, str]) -> Dict[str, str]:
        """Adds authentication details to the request parameters."""
        if self.api_key:
            params["api_key"] = self.api_key
        # NCBI recommends providing an email even if using an API key
        if self.email:
             params["email"] = self.email
        return params

    @ultra_cache(expiry_seconds=86400 * 7)  # Cache for 7 days
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def fetch_details(self, pmids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetches publication details for a list of PMIDs.
        Results are cached automatically.
        """
        if not pmids:
            return {}

        logger.info(f"Fetching details for {len(pmids)} PMIDs from PubMed (batched).")

        BATCH = 200  # Avoid long URLs
        session = await self._get_session()
        results: Dict[str, Dict[str, Any]] = {}

        for i in range(0, len(pmids), BATCH):
            chunk = pmids[i:i+BATCH]
            params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"}
            try:
                async with session.post(f"{self.BASE_URL}/efetch.fcgi", params=self._add_auth(params)) as response:
                    response.raise_for_status()
                    text = await response.text()

                root = ET.fromstring(text)
                for article_node in root.findall(".//PubmedArticle"):
                    try:
                        pmid = article_node.findtext(".//PMID")
                        if not pmid:
                            continue

                        title = article_node.findtext(".//ArticleTitle") or ""

                        abstract_nodes = article_node.findall(".//Abstract/AbstractText")
                        abstract = "\n".join([node.text for node in abstract_nodes if node.text])

                        authors_list = [
                            f"{author.findtext('LastName', '')} {author.findtext('Initials', '')}".strip()
                            for author in article_node.findall(".//Author")
                        ]

                        journal = article_node.findtext(".//Title") or ""
                        pub_year = article_node.findtext(".//PubDate/Year") or ""

                        results[pmid] = {
                            "title": title,
                            "abstract": abstract,
                            "authors": authors_list,
                            "journal": journal,
                            "publication_date": pub_year,
                        }
                    except Exception as parse_err:
                        logger.error("Error parsing article PMID %s: %s", pmid, parse_err)
            except Exception as fetch_err:
                logger.error("Failed to fetch batch starting at %s: %s", i, fetch_err)

        await self.close_session()
        return results

    async def search(self, query: str, max_results: int = 10, **kwargs) -> EvidenceCorpus:
        """
        Search PubMed and return results as EvidenceCorpus.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            EvidenceCorpus containing the search results
        """
        try:
            # Step 1: Search for PMIDs
            pmids = await self.search_pmids(query, max_results)
            
            if not pmids:
                logger.warning(f"No PMIDs found for query: {query}")
                return EvidenceCorpus(query=query, sources=[])
            
            # Step 2: Fetch details for PMIDs
            details = await self.fetch_details(pmids)
            
            # Step 3: Convert to EvidenceSource objects
            sources = []
            for pmid, data in details.items():
                if data.get('title') and data.get('abstract'):
                    source = EvidenceSource(
                        id=f"pubmed_{pmid}",
                        title=data['title'],
                        summary=data['abstract'][:500] + "..." if len(data['abstract']) > 500 else data['abstract'],
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        source_name="PubMed",
                        authors=data.get('authors', []),
                        publication_date=data.get('publication_date'),
                        journal=data.get('journal'),
                        confidence_score=0.8  # High confidence for PubMed sources
                    )
                    sources.append(source)
            
            logger.info(f"PubMed search found {len(sources)} sources for query: {query[:50]}...")
            return EvidenceCorpus(
                query=query, 
                sources=sources,
                total_results=len(sources),
                filtered_results=len(sources)
            )
            
        except Exception as e:
            logger.error(f"PubMed search failed for query '{query}': {e}")
            return EvidenceCorpus(query=query, sources=[])

    # Rename the old search method to search_pmids
    async def search_pmids(self, query: str, max_results: int = 20) -> List[str]:
        """Search for PMIDs only (original search method renamed)"""
        logger.info(f"Searching PubMed for: '{query[:50]}...'")
        
        session = await self._get_session()
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }
        
        async with session.get(f"{self.BASE_URL}/esearch.fcgi", params=self._add_auth(params)) as response:
            response.raise_for_status()
            data = await response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            
            return pmids
