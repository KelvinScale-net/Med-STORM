from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, AsyncGenerator, Optional, Union, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from med_storm.models.evidence import EvidenceCorpus, EvidenceSource

class KnowledgeConnector(ABC):
    """Abstract Base Class for all knowledge connectors."""

    @abstractmethod
    async def search(
        self, 
        query: str, 
        **kwargs: Any
    ) -> 'EvidenceCorpus':
        """
        Search for evidence related to a query.
        
        Args:
            query: The search query string.
            **kwargs: Additional search parameters specific to the connector.
            
        Returns:
            EvidenceCorpus: A corpus containing the search results.
        """
        pass
        
    async def search_stream(
        self,
        query: str,
        **kwargs: Any
    ) -> AsyncGenerator[Union['EvidenceSource', List['EvidenceSource']], None]:
        """
        Stream search results as they become available.
        
        This is an optional method that connectors can implement to provide
        streaming results. The default implementation falls back to the regular
        search method and yields a single result.
        
        Args:
            query: The search query string.
            **kwargs: Additional search parameters specific to the connector.
            
        Yields:
            Dictionaries containing partial results or status updates.
            The final yield will be an EvidenceCorpus object.
        """
        # Default implementation falls back to regular search
        result = await self.search(query, **kwargs)
        yield {"status": "search_complete", "message": "Using non-streaming search"}
        yield result
