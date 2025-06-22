import asyncio
import os

# Adjust path to import from the project's source
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from med_storm.connectors.pubmed import PubMedConnector

async def main():
    """Main function to test the PubMedConnector."""
    print("Testing PubMed Connector...")
    connector = PubMedConnector(max_results=3)
    
    # A classic medical test query
    query = "Aspirin for primary prevention of cardiovascular disease"
    print(f"Searching for: '{query}'")
    
    try:
        evidence_corpus = await connector.search(query)
        
        if not evidence_corpus.sources:
            print("\nNo results found. This might be due to network issues or no matching articles.")
            return

        print(f"\nFound {len(evidence_corpus.sources)} articles for query: '{evidence_corpus.query}'")
        print("-" * 50)
        
        for i, source in enumerate(evidence_corpus.sources, 1):
            print(f"\nArticle {i}:")
            print(f"  Title: {source.title}")
            print(f"  URL: {source.url}")
            print(f"  Published: {source.published_date}")
            print(f"  Authors: {', '.join(source.authors)}")
            print(f"  Summary: {source.summary[:250]}...") # Print first 250 chars of summary
        print("\nPubMed Connector test successful!")

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
        print("Please ensure you have an internet connection and that the NCBI services are available.")
        print("If you are running many requests, you may need a PUBMED_API_KEY in your .env file.")

if __name__ == "__main__":
    # Ensure you have installed the requirements: pip install -r requirements.txt
    # You might need to create a .env file if you want to use an API key.
    asyncio.run(main())
