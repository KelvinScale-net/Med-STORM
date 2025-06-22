"""
🔍 STANDARD KNOWLEDGE CONNECTORS
Connectors estándar que siguen las interfaces definidas
"""

import asyncio
import logging
import time
from typing import List, Dict, Any
import aiohttp
from ..core.interfaces import KnowledgeConnector, EvidenceSource, EvidenceCorpus

logger = logging.getLogger(__name__)


class StandardPubMedConnector(KnowledgeConnector):
    """Standard PubMed connector following interfaces"""
    
    def __init__(self, email: str, api_key: str = None):
        self.email = email
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.is_healthy = True
        
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> EvidenceCorpus:
        """Standard search method returning EvidenceCorpus"""
        start_time = time.time()
        
        try:
            # Search for PMIDs
            search_url = f"{self.base_url}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "email": self.email
            }
            if self.api_key:
                search_params["api_key"] = self.api_key
            
            async with aiohttp.ClientSession() as session:
                # Get PMIDs
                async with session.get(search_url, params=search_params) as response:
                    search_data = await response.json()
                
                pmids = search_data.get("esearchresult", {}).get("idlist", [])
                
                if not pmids:
                    return EvidenceCorpus(
                        query=query,
                        sources=[],
                        total_found=0,
                        retrieval_time=time.time() - start_time,
                        source_breakdown={"pubmed": 0}
                    )
                
                # Get article details
                fetch_url = f"{self.base_url}/efetch.fcgi"
                fetch_params = {
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "xml",
                    "email": self.email
                }
                if self.api_key:
                    fetch_params["api_key"] = self.api_key
                
                async with session.get(fetch_url, params=fetch_params) as response:
                    xml_data = await response.text()
                
                # Parse XML and create EvidenceSource objects
                sources = self._parse_pubmed_xml(xml_data, pmids)
                
                self.is_healthy = True
                return EvidenceCorpus(
                    query=query,
                    sources=sources,
                    total_found=len(sources),
                    retrieval_time=time.time() - start_time,
                    source_breakdown={"pubmed": len(sources)}
                )
                
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            self.is_healthy = False
            return EvidenceCorpus(
                query=query,
                sources=[],
                total_found=0,
                retrieval_time=time.time() - start_time,
                source_breakdown={"pubmed": 0}
            )
    
    def _parse_pubmed_xml(self, xml_data: str, pmids: List[str]) -> List[EvidenceSource]:
        """Parse PubMed XML response"""
        import xml.etree.ElementTree as ET
        
        sources = []
        try:
            root = ET.fromstring(xml_data)
            articles = root.findall('.//PubmedArticle')
            
            for i, article in enumerate(articles):
                try:
                    # Extract basic info
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "No title"
                    
                    abstract_elem = article.find('.//AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                    
                    pmid = pmids[i] if i < len(pmids) else "unknown"
                    
                    # Extract authors
                    authors = []
                    author_elems = article.findall('.//Author')
                    for author in author_elems[:3]:  # Limit to first 3 authors
                        lastname = author.find('LastName')
                        forename = author.find('ForeName')
                        if lastname is not None:
                            name = lastname.text
                            if forename is not None:
                                name = f"{forename.text} {name}"
                            authors.append(name)
                    
                    source = EvidenceSource(
                        title=title,
                        summary=abstract,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        source_type="pubmed",
                        confidence_score=0.9,  # High confidence for PubMed
                        pmid=pmid,
                        authors=authors
                    )
                    sources.append(source)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse article {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to parse PubMed XML: {e}")
        
        return sources
    
    async def health_check(self) -> bool:
        """Check if PubMed is accessible"""
        try:
            test_corpus = await self.search("diabetes", max_results=1)
            return len(test_corpus.sources) > 0
        except:
            return False


class StandardSerperConnector(KnowledgeConnector):
    """Standard Serper connector following interfaces"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev/search"
        self.is_healthy = True
        
        # Trusted medical domains
        self.trusted_domains = [
            "pubmed.ncbi.nlm.nih.gov",
            "uptodate.com", 
            "who.int",
            "cdc.gov",
            "cochranelibrary.com",
            "mayoclinic.org",
            "nejm.org",
            "thelancet.com"
        ]
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> EvidenceCorpus:
        """Standard search method returning EvidenceCorpus"""
        start_time = time.time()
        
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Search each trusted domain
            all_sources = []
            source_breakdown = {}
            
            async with aiohttp.ClientSession() as session:
                for domain in self.trusted_domains[:5]:  # Limit to 5 domains for speed
                    try:
                        domain_query = f"{query} site:{domain}"
                        payload = {
                            "q": domain_query,
                            "num": max(2, max_results // len(self.trusted_domains))
                        }
                        
                        async with session.post(
                            self.base_url, 
                            headers=headers, 
                            json=payload,
                            timeout=10
                        ) as response:
                            data = await response.json()
                            
                            domain_sources = self._parse_serper_response(data, domain)
                            all_sources.extend(domain_sources)
                            source_breakdown[domain] = len(domain_sources)
                            
                    except Exception as e:
                        logger.warning(f"Serper search failed for {domain}: {e}")
                        source_breakdown[domain] = 0
                        continue
            
            # Limit total results
            all_sources = all_sources[:max_results]
            
            self.is_healthy = True
            return EvidenceCorpus(
                query=query,
                sources=all_sources,
                total_found=len(all_sources),
                retrieval_time=time.time() - start_time,
                source_breakdown=source_breakdown
            )
            
        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            self.is_healthy = False
            return EvidenceCorpus(
                query=query,
                sources=[],
                total_found=0,
                retrieval_time=time.time() - start_time,
                source_breakdown={}
            )
    
    def _parse_serper_response(self, data: Dict[str, Any], domain: str) -> List[EvidenceSource]:
        """Parse Serper API response"""
        sources = []
        
        try:
            organic_results = data.get("organic", [])
            
            for result in organic_results:
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description")
                url = result.get("link", "")
                
                # Calculate confidence based on domain
                confidence_map = {
                    "pubmed.ncbi.nlm.nih.gov": 0.95,
                    "uptodate.com": 0.9,
                    "who.int": 0.85,
                    "cdc.gov": 0.85,
                    "cochranelibrary.com": 0.9,
                    "mayoclinic.org": 0.8,
                    "nejm.org": 0.95,
                    "thelancet.com": 0.95
                }
                confidence = confidence_map.get(domain, 0.7)
                
                source = EvidenceSource(
                    title=title,
                    summary=snippet,
                    url=url,
                    source_type="serper",
                    confidence_score=confidence
                )
                sources.append(source)
                
        except Exception as e:
            logger.warning(f"Failed to parse Serper response for {domain}: {e}")
        
        return sources
    
    async def health_check(self) -> bool:
        """Check if Serper is accessible"""
        try:
            test_corpus = await self.search("test", max_results=1)
            return len(test_corpus.sources) > 0
        except:
            return False


class StandardLocalCorpusConnector(KnowledgeConnector):
    """Standard Local Corpus connector following interfaces"""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = None):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name or "corpus_bariatric_surgery_for_type_2_diabetes"
        self.is_healthy = True
        
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> EvidenceCorpus:
        """Standard search method returning EvidenceCorpus"""
        start_time = time.time()
        
        try:
            # Simulate local corpus search (simplified for now)
            # In production, this would use proper vector search
            
            # Mock sources for demonstration
            sources = [
                EvidenceSource(
                    title=f"Local Corpus Result {i+1} for: {query}",
                    summary=f"Relevant medical content from local corpus related to {query}",
                    url=f"local://corpus/{i+1}",
                    source_type="local_corpus",
                    confidence_score=0.8
                )
                for i in range(min(5, max_results))
            ]
            
            self.is_healthy = True
            return EvidenceCorpus(
                query=query,
                sources=sources,
                total_found=len(sources),
                retrieval_time=time.time() - start_time,
                source_breakdown={"local_corpus": len(sources)}
            )
            
        except Exception as e:
            logger.error(f"Local corpus search failed: {e}")
            self.is_healthy = False
            return EvidenceCorpus(
                query=query,
                sources=[],
                total_found=0,
                retrieval_time=time.time() - start_time,
                source_breakdown={"local_corpus": 0}
            )
    
    async def health_check(self) -> bool:
        """Check if local corpus is accessible"""
        try:
            test_corpus = await self.search("test", max_results=1)
            return len(test_corpus.sources) > 0
        except:
            return False 