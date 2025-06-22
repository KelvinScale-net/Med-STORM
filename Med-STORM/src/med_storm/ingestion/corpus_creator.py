# src/med_storm/ingestion/corpus_creator.py

import logging
from typing import List, Dict, Any, Optional
import uuid

import pubmed_parser as pp
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from med_storm.connectors.pubmed import PubMedConnector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorpusCreator:
    """
    Handles the creation of a research corpus by fetching data from PubMed,
    processing it, generating vector embeddings, and storing them in Qdrant.
    """
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        pubmed_email: Optional[str] = None,
    ):
        """
        Initializes the CorpusCreator.

        Args:
            qdrant_host (str): The host of the Qdrant instance.
            qdrant_port (int): The port of the Qdrant instance.
            embedding_model_name (str): The name of the sentence-transformer model to use.
            pubmed_email (Optional[str]): The email to use for the PubMed API.
        """
        logger.info("Initializing CorpusCreator...")
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.info(f"Loading sentence transformer model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.pubmed_connector = PubMedConnector(email=pubmed_email)
        logger.info("CorpusCreator initialized successfully.")

    async def create_corpus(
        self,
        topic: str,
        collection_name: str,
        max_articles: int = 100
    ):
        """
        Creates and populates a new collection in Qdrant for a given research topic.

        Args:
            topic (str): The research topic to build the corpus for.
            collection_name (str): The name of the collection to create in Qdrant.
            max_articles (int): The maximum number of articles to fetch from PubMed.
        """
        logger.info(f"Starting corpus creation for topic: '{topic}' in collection: '{collection_name}'")

        # 1. Create Qdrant collection
        self._create_qdrant_collection(collection_name)

        # 2. Fetch article PMIDs from PubMed
        logger.info(f"Searching PubMed for articles related to '{topic}'...")
        pmids = await self.pubmed_connector.search(topic, max_results=max_articles)
        if not pmids:
            logger.warning(f"No articles found on PubMed for the topic: {topic}")
            return

        logger.info(f"Found {len(pmids)} potential articles. Fetching details...")

        # 3. Fetch article details
        # The pubmed_parser library works with file paths, so we download the abstracts
        # This part will be fleshed out. For now, we'll assume we get dicts.
        # In a real implementation, we might use a temporary file.
        # For now, let's simulate getting article data.
        articles = await self._fetch_article_details(pmids)
        if not articles:
            logger.warning("Could not fetch details for the found PMIDs.")
            return

        # 4. Generate embeddings and prepare for upsert
        points_to_upsert = self._prepare_qdrant_points(articles)
        if not points_to_upsert:
            logger.warning("No valid articles could be processed for Qdrant upsert.")
            return

        # 5. Upsert data into Qdrant
        logger.info(f"Upserting {len(points_to_upsert)} points into Qdrant collection '{collection_name}'...")
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points_to_upsert,
            wait=True
        )
        logger.info("Corpus creation complete.")

    def _create_qdrant_collection(self, collection_name: str):
        """
        Creates a new collection in Qdrant if it doesn't already exist.
        """
        vector_size = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Ensuring Qdrant collection '{collection_name}' exists with vector size {vector_size}.")
        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    async def _fetch_article_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetches the full details for a list of PubMed IDs.
        
        This is a placeholder and will need to be implemented robustly.
        It should ideally use the pubmed_parser library after downloading the abstracts.
        """
        # Placeholder implementation
        logger.info(f"Fetching details for {len(pmids)} PMIDs (placeholder implementation)...")
        # In a real scenario, we'd use Biopython or pubmed_parser to get full records.
        # For now, we simulate this by calling our existing connector.
        articles = await self.pubmed_connector.fetch_details(pmids)
        
        # We need to structure it in a way pubmed_parser would have
        structured_articles = []
        for pmid, article_data in articles.items():
            if article_data and article_data.get("abstract"):
                 structured_articles.append({
                    "pmid": pmid,
                    "title": article_data.get("title", ""),
                    "abstract": article_data.get("abstract", ""),
                    "journal": article_data.get("journal", ""),
                    "authors": article_data.get("authors", [])
                 })
        return structured_articles


    def _prepare_qdrant_points(self, articles: List[Dict[str, Any]]) -> List[models.PointStruct]:
        """
        Generates embeddings for articles and prepares them as Qdrant PointStructs.
        """
        abstracts = [article['abstract'] for article in articles if article.get('abstract')]
        if not abstracts:
            logger.warning("No abstracts found in the provided articles to generate embeddings.")
            return []

        logger.info(f"Generating embeddings for {len(abstracts)} abstracts...")
        embeddings = self.embedding_model.encode(abstracts, show_progress_bar=True)

        points = []
        for i, article in enumerate(articles):
            if article.get('abstract'):
                # Generate a UUID for the point ID
                point_id = str(uuid.uuid4())

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings[i].tolist(),
                        payload={
                            "pmid": article.get("pmid", ""), # Store original PMID here
                            "title": article.get("title", ""),
                            "journal": article.get("journal", ""),
                            "authors": ", ".join(article.get("authors", [])),
                            "text": article.get("abstract", "") # The text we searched on
                        }
                    )
                )
        return points

async def close_connections():
    # Placeholder for closing any open connections if needed
    pass 