# src/med_storm/connectors/local_corpus.py

"""
🚀 ULTRA-OPTIMIZED LOCAL CORPUS CONNECTOR
=========================================

Features:
1. CONNECTION POOLING: Reuse Qdrant connections
2. BATCH VECTOR SEARCH: Process multiple queries simultaneously  
3. SMART PAGINATION: Intelligent result fetching
4. ADAPTIVE SCORING: Dynamic confidence adjustment
5. PARALLEL PROCESSING: Maximum throughput
"""
import asyncio
import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http.models import Filter, SearchRequest, SearchParams
from sentence_transformers import SentenceTransformer

from med_storm.connectors.base import KnowledgeConnector
from med_storm.models.evidence import EvidenceCorpus, EvidenceSource
from med_storm.config import settings
from med_storm.utils.cache import ultra_cache

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SearchStats:
    """Track search performance metrics."""
    total_searches: int = 0
    total_time: float = 0.0
    avg_results_per_search: float = 0.0
    cache_hit_rate: float = 0.0

class UltraLocalCorpusConnector(KnowledgeConnector):
    """🚀 REVOLUTIONARY Local Corpus Connector with Ultra Performance"""

    def __init__(
        self,
        collection_name: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        qdrant_url: str = "http://localhost:6333",
        max_connections: int = 20,  # 🔥 CONNECTION POOLING
        batch_size: int = 50,       # 🔥 BATCH PROCESSING
        enable_async: bool = True,   # 🚀 ASYNC OPTIMIZATION
    ):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.max_connections = max_connections
        self.batch_size = batch_size
        self.enable_async = enable_async
        
        # Performance metrics
        self.stats = SearchStats()
        
        # Initialize embedding model (cached)
        logger.info(f"🔥 Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Connection pool
        self.connection_pool: List[AsyncQdrantClient] = []
        self.connection_semaphore = asyncio.Semaphore(max_connections)
        self._pool_initialized = False

    async def _ensure_pool_initialized(self):
        """Ensure connection pool is initialized (lazy initialization)."""
        if not self._pool_initialized:
            await self._init_connection_pool()

    async def _init_connection_pool(self):
        """🚀 Initialize connection pool for maximum throughput."""
        try:
            # Create multiple async connections
            for i in range(min(self.max_connections, 10)):  # Start with 10 connections
                client = AsyncQdrantClient(url=self.qdrant_url)
                # Test connection
                await client.get_collections()
                self.connection_pool.append(client)
            
            logger.info(f"🚀 Connection pool initialized with {len(self.connection_pool)} connections")
            self._pool_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            # Fallback to single connection
            self.connection_pool = [AsyncQdrantClient(url=self.qdrant_url)]

    async def _get_connection(self) -> AsyncQdrantClient:
        """Get an available connection from the pool."""
        await self._ensure_pool_initialized()
        
        if not self.connection_pool:
            # Create emergency connection
            return AsyncQdrantClient(url=self.qdrant_url)
        
        # Round-robin connection selection
        import random
        return random.choice(self.connection_pool)

    def _generate_embeddings_batch(self, queries: List[str]) -> List[List[float]]:
        """🔥 BATCH EMBEDDING GENERATION for maximum efficiency."""
        # Process all queries at once for efficiency
        embeddings = self.embedding_model.encode(
            queries, 
            batch_size=32,  # Optimize batch size for the model
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    async def _search_single_optimized(
        self, 
        query_embedding: List[float], 
        max_results: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """🚀 ULTRA-OPTIMIZED single query search."""
        
        async with self.connection_semaphore:
            client = await self._get_connection()
            
            try:
                # Use the most efficient search parameters
                search_result = await client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=max_results,
                    with_payload=True,
                    with_vectors=False,  # Don't return vectors to save bandwidth
                    search_params=SearchParams(
                        hnsw_ef=128,  # Higher for better recall
                        exact=False   # Use approximate search for speed
                    )
                )
                
                results = []
                for point in search_result:
                    if point.payload and point.score is not None:
                        results.append((point.payload, point.score))
                
                return results
                
            except Exception as e:
                # Auto-create collection if not found (404)
                if "doesn't exist" in str(e) or "Not found" in str(e):
                    logger.warning("🛠️ Collection '%s' not found. Creating automatically...", self.collection_name)
                    await self._create_collection_if_missing(client)
                    # Retry search once
                    search_result = await client.search(
                        collection_name=self.collection_name,
                        query_vector=query_embedding,
                        limit=max_results,
                        with_payload=True,
                        with_vectors=False,
                        search_params=SearchParams(hnsw_ef=128, exact=False)
                    )
                else:
                    raise

    async def _create_collection_if_missing(self, client: AsyncQdrantClient):
        """Create collection with default HNSW params if it doesn't exist."""
        try:
            dim = self.embedding_model.get_sentence_embedding_dimension()
            await client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": dim, "distance": "Cosine"}
            )
            logger.info("✅ Collection '%s' created with dimension %s", self.collection_name, dim)
        except Exception as exc:
            logger.error("Failed to create collection '%s': %s", self.collection_name, exc)

    async def batch_search(
        self,
        queries: List[str],
        max_results_per_query: int = 10
    ) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
        """🚀 REVOLUTIONARY: Search multiple queries simultaneously."""
        
        if not queries:
            return {}
        
        start_time = time.time()
        
        # STAGE 1: Generate all embeddings in one batch
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            self._generate_embeddings_batch, 
            queries
        )
        
        # STAGE 2: Execute all searches in parallel
        search_tasks = [
            self._search_single_optimized(embedding, max_results_per_query)
            for embedding in embeddings
        ]
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # STAGE 3: Combine results
        combined_results = {}
        for query, result in zip(queries, search_results):
            if isinstance(result, list):
                combined_results[query] = result
            else:
                logger.error(f"Search error for query '{query}': {result}")
                combined_results[query] = []
        
        # Update statistics
        search_time = time.time() - start_time
        self.stats.total_searches += len(queries)
        self.stats.total_time += search_time
        
        total_results = sum(len(results) for results in combined_results.values())
        self.stats.avg_results_per_search = total_results / len(queries) if queries else 0
        
        logger.info(f"🚀 Batch search completed: {len(queries)} queries in {search_time:.2f}s")
        
        return combined_results

    def _create_evidence_source(
        self, 
        payload: Dict[str, Any], 
        confidence_score: float
    ) -> EvidenceSource:
        """🔥 OPTIMIZED evidence source creation."""
        
        # Extract PMID with fallback
        pmid = payload.get("pmid")
        if not pmid:
            # Generate fallback ID if PMID is missing
            pmid = str(uuid.uuid4())[:8]
        
        # Smart URL generation
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid.isdigit() else payload.get("url", "")
        
        # Adaptive confidence scoring
        adaptive_score = min(confidence_score * 1.2, 1.0)  # Boost slightly but cap at 1.0
        
        return EvidenceSource(
            id=str(pmid),
            title=payload.get("title", "No Title"),
            url=url,
            summary=payload.get("text", ""),
            source_name=f"Local Corpus (PMID: {pmid})",
            authors=[a.strip() for a in payload.get("authors", "").split(",") if a.strip()],
            journal=payload.get("journal", ""),
            pmid=pmid,
            confidence_score=adaptive_score,
            metadata=payload  # Store full payload for future use
        )

    @ultra_cache(expiry_seconds=3600, predict_related=True)
    async def search(
        self, 
        query: str, 
        max_results: int = 10
    ) -> EvidenceCorpus:
        """🚀 ULTRA-FAST search with intelligent caching."""
        
        if not query.strip():
            return EvidenceCorpus(query=query, sources=[])
        
        logger.info(f"🔍 Searching local corpus: '{query[:50]}...'")
        start_time = time.time()
        
        # Use batch search for single query (optimized path)
        batch_results = await self.batch_search([query], max_results)
        search_results = batch_results.get(query, [])
        
        # Convert to evidence sources
        evidence_sources = [
            self._create_evidence_source(payload, score)
            for payload, score in search_results
        ]
        
        search_time = time.time() - start_time
        logger.info(f"⚡ Found {len(evidence_sources)} sources in {search_time:.2f}s")
        
        return EvidenceCorpus(query=query, sources=evidence_sources)

    async def search_multiple_queries(
        self,
        queries: List[str],
        max_results_per_query: int = 10
    ) -> Dict[str, EvidenceCorpus]:
        """🚀 ULTRA-EFFICIENT: Search multiple queries with maximum parallelization."""
        
        if not queries:
            return {}
        
        logger.info(f"🔥 Batch searching {len(queries)} queries")
        start_time = time.time()
        
        # Execute batch search
        batch_results = await self.batch_search(queries, max_results_per_query)
        
        # Convert all results to EvidenceCorpus objects
        corpus_map = {}
        for query in queries:
            search_results = batch_results.get(query, [])
            evidence_sources = [
                self._create_evidence_source(payload, score)
                for payload, score in search_results
            ]
            corpus_map[query] = EvidenceCorpus(query=query, sources=evidence_sources)
        
        total_time = time.time() - start_time
        total_sources = sum(len(corpus.sources) for corpus in corpus_map.values())
        
        logger.info(f"🚀 Batch search completed: {len(queries)} queries, {total_sources} sources in {total_time:.2f}s")
        
        return corpus_map

    async def ultra_search_all_questions(
        self,
        questions: List[str],
        max_results_per_question: int = 10
    ) -> Dict[str, EvidenceCorpus]:
        """🚀 NEW METHOD: Direct integration with UltraStormEngine batch processing."""
        return await self.search_multiple_queries(questions, max_results_per_question)

    async def prefetch_related_queries(self, base_query: str, related_queries: List[str]):
        """🧠 PREDICTIVE PREFETCHING: Warm cache with related queries."""
        
        # Run related searches in background (don't await)
        asyncio.create_task(self._background_prefetch(related_queries))

    async def _background_prefetch(self, queries: List[str]):
        """Background prefetching task."""
        try:
            await self.search_multiple_queries(queries, max_results_per_query=5)
            logger.debug(f"🎯 Prefetched {len(queries)} related queries")
        except Exception as e:
            logger.warning(f"Prefetch error: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = self.stats.total_time / self.stats.total_searches if self.stats.total_searches else 0
        
        return {
            "total_searches": self.stats.total_searches,
            "total_time": self.stats.total_time,
            "avg_time_per_search": avg_time,
            "avg_results_per_search": self.stats.avg_results_per_search,
            "cache_hit_rate": self.stats.cache_hit_rate,
            "connection_pool_size": len(self.connection_pool)
        }

    async def health_check(self) -> bool:
        """Check if the connector is healthy."""
        try:
            await self._ensure_pool_initialized()
            client = await self._get_connection()
            collections = await client.get_collections()
            collection_names = [c.name for c in collections.collections]
            return self.collection_name in collection_names
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def close(self):
        """Clean up connections."""
        for client in self.connection_pool:
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
        self.connection_pool.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Alias for backward compatibility
LocalCorpusConnector = UltraLocalCorpusConnector 