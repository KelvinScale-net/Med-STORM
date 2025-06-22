# scripts/create_corpus.py

import asyncio
import argparse
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from med_storm.ingestion.corpus_creator import CorpusCreator
from med_storm.config import settings

async def main():
    """
    Main function to run the corpus creation process.
    """
    parser = argparse.ArgumentParser(description="Create a research corpus in Qdrant from PubMed articles.")
    parser.add_argument("topic", type=str, help="The research topic to build the corpus for.")
    parser.add_argument(
        "--collection_name",
        type=str,
        default=None,
        help="The name of the Qdrant collection. Defaults to a sanitized version of the topic.",
    )
    parser.add_argument(
        "--max_articles",
        type=int,
        default=200,
        help="The maximum number of articles to fetch from PubMed.",
    )
    args = parser.parse_args()

    # If no collection name is provided, create one from the topic
    collection_name = args.collection_name
    if not collection_name:
        collection_name = args.topic.lower().replace(" ", "_").replace("-", "_")
        collection_name = f"corpus_{collection_name}"

    # Initialize the CorpusCreator
    creator = CorpusCreator(
        qdrant_host=settings.QDRANT_HOST,
        qdrant_port=settings.QDRANT_PORT,
        pubmed_email=settings.PUBMED_EMAIL,
    )

    # Run the corpus creation process
    await creator.create_corpus(
        topic=args.topic,
        collection_name=collection_name,
        max_articles=args.max_articles,
    )

if __name__ == "__main__":
    asyncio.run(main()) 