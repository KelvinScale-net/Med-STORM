import asyncio
import aiohttp
import logging
from typing import List, Optional

from med_storm.connectors.base import KnowledgeConnector
from med_storm.models.evidence import EvidenceCorpus, EvidenceSource
from med_storm.config import settings
from med_storm.utils.cache import ultra_cache

# Configure logging
logger = logging.getLogger(__name__)

class SerperConnector(KnowledgeConnector):
    """A knowledge connector for targeted web searches using the Serper API (google.serper.dev)."""

    def __init__(self, api_key: Optional[str] = None, trusted_domains: Optional[List[str]] = None):
        self.api_key = api_key or settings.SERPER_API_KEY
        if not self.api_key:
            raise ValueError("SERPER_API_KEY is not set in the environment variables.")
        self.base_url = "https://google.serper.dev"
        self.trusted_domains = trusted_domains or [
            "pubmed.ncbi.nlm.nih.gov",
            "cochranelibrary.com", 
            "uptodate.com",
            "who.int",
            "cdc.gov",
            "fda.gov",
            "nejm.org",
            "thelancet.com",
            "jamanetwork.com",
            "bmj.com",
            "nature.com",
            "sciencedirect.com"
        ]
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close_session(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("SerperConnector session closed.")

    async def _search_domain(self, query: str, domain: str, max_results: int = 5) -> List[EvidenceSource]:
        """Search a specific domain using Serper API."""
        search_query = f"site:{domain} {query}"
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": search_query,
            "num": max_results
        }
        
        try:
            session = await self._get_session()
            async with session.post(f"{self.base_url}/search", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    sources = []
                    
                    # Process organic results
                    for result in data.get("organic", [])[:max_results]:
                        if result.get("link"):
                            source = EvidenceSource(
                                id=f"serper_{len(sources)}",
                                title=result.get("title", "No title found"),
                                summary=result.get("snippet", "No summary available."),
                                url=result.get("link"),
                                source_name=f"Serper-{domain}",
                                confidence_score=0.7,
                                authors=[]
                            )
                            sources.append(source)
                    
                    logger.info(f"✅ Serper found {len(sources)} sources for domain {domain}")
                    return sources
                else:
                    logger.warning(f"⚠️ Serper API error for domain {domain}: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Serper search failed for domain {domain}: {e}")
            return []

    @ultra_cache(expiry_seconds=3600)  # Cache for 1 hour
    async def search(self, query: str, max_results: int = 10, **kwargs) -> EvidenceCorpus:
        """
        Search the web using Serper's Google Search API, filtered by trusted domains.
        
        Args:
            query: Search query string
            max_results: Maximum results per domain
            
        Returns:
            EvidenceCorpus containing search results
        """
        logger.info(f"🔍 Serper searching for: '{query[:50]}...'")
        
        all_sources: List[EvidenceSource] = []
        
        if not self.trusted_domains:
            logger.warning("No trusted domains configured for Serper search")
            return EvidenceCorpus(query=query, sources=[])

        # Search each domain concurrently
        tasks = [
            self._search_domain(query, domain, max_results)
            for domain in self.trusted_domains[:5]  # Limit to first 5 domains to avoid rate limits
        ]

        try:
            domain_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(domain_results):
                if isinstance(result, Exception):
                    logger.error(f"Domain search failed for {self.trusted_domains[i]}: {result}")
                else:
                    all_sources.extend(result)
            
            # Remove duplicates based on URL
            unique_sources = []
            seen_urls = set()
            for source in all_sources:
                if source.url not in seen_urls:
                    unique_sources.append(source)
                    seen_urls.add(source.url)
            
            logger.info(f"🌐 Serper search completed: {len(unique_sources)} unique sources found")
            
            return EvidenceCorpus(
                query=query,
                sources=unique_sources,
                total_results=len(all_sources),
                filtered_results=len(unique_sources)
            )
            
        except Exception as e:
            logger.error(f"❌ Serper search failed: {e}")
            return EvidenceCorpus(query=query, sources=[])
        finally:
            await self.close_session()
