import asyncio
import aiohttp
import re
from typing import List
from bs4 import BeautifulSoup
from duckduckgo_search import AsyncDDGS

from med_storm.connectors.base import KnowledgeConnector
from med_storm.models.evidence import EvidenceCorpus, EvidenceSource

class WebConnector(KnowledgeConnector):
    """A knowledge connector for targeted web searches on trusted domains."""

    def __init__(self, trusted_domains: List[str]):
        self.trusted_domains = trusted_domains
        self._semaphore = asyncio.Semaphore(1)

    def _simplify_query(self, query: str) -> str:
        """Simplifies a complex PubMed-style query to a keyword-based query."""
        # Remove MeSH tags and other bracketed terms
        simplified_query = re.sub(r'\[.*?\]', '', query)
        # Remove boolean operators
        simplified_query = re.sub(r'\b(AND|OR|NOT)\b', '', simplified_query, flags=re.IGNORECASE)
        # Remove quotes
        simplified_query = simplified_query.replace('"', '')
        # Collapse multiple spaces into one
        simplified_query = re.sub(r'\s+', ' ', simplified_query).strip()
        return simplified_query

    async def search(
        self, query: str, max_results_per_domain: int = 3
    ) -> EvidenceCorpus:
        """
        Searches trusted web domains for a given query, respecting rate limits.
        """
        simplified_query = self._simplify_query(query)
        async with self._semaphore:
            all_sources: List[EvidenceSource] = []

            async def fetch_and_parse(session, result, domain):
                """Fetches a URL and parses it for title and summary."""
                try:
                    href = result.get('href')
                    if not href:
                        return None
                    async with session.get(href, timeout=30) as response:
                        if response.status == 200:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            title = soup.title.string if soup.title else result.get('title', 'No title found')
                            summary = result.get('body', 'No summary available.')
                            return EvidenceSource(
                                title=title,
                                summary=summary,
                                url=href,
                                source_name=domain,
                                confidence_score=0.6,  # Medium confidence for general web sources
                                authors=[]
                            )
                except Exception as e:
                    pass
                return None

            async with aiohttp.ClientSession() as session, AsyncDDGS(timeout=30) as ddgs:
                for domain in self.trusted_domains:
                    try:
                        search_query = f"site:{domain} {simplified_query}"
                        
                        tasks = []
                        search_results = await ddgs.text(search_query, max_results=max_results_per_domain)
                        for result in search_results:
                            tasks.append(fetch_and_parse(session, result, domain))
                        
                        sources = await asyncio.gather(*tasks)
                        all_sources.extend([s for s in sources if s])
                        
                        await asyncio.sleep(5)

                    except Exception as e:
                        print(f"Error searching on domain {domain}: {e}")
            
            return EvidenceCorpus(query=query, sources=all_sources)
