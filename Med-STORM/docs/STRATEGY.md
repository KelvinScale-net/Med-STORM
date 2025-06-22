# Med-STORM: Architectural Evolution & Strategy

This document outlines the strategic pivot for the Med-STORM project, moving away from a live API-centric model to a robust, local-first Retrieval-Augmented Generation (RAG) architecture. This shift is designed to address the core issues of speed, reliability, and depth of research.

## 1. Core Problem Analysis

The previous architecture suffered from several critical flaws:

-   **High Latency:** Each generated research question triggered a new, live API call to PubMed. Network latency and API rate limits created a significant bottleneck, making the research process impractically slow.
-   **Reliability Issues:** Dependence on external APIs introduced fragility. We frequently encountered `invalid predicate` errors and other network-related failures that were difficult to control or debug, halting the entire process.
-   **Shallow Search:** The "one-question, one-search" model prevented us from building a deep, contextual understanding of the topic. The system couldn't see the forest for the trees, as it never had a comprehensive corpus of information to draw from.

## 2. The New Vision: Local-First RAG Architecture

Inspired by best-in-class frameworks like `STORM`, `Med-PRM`, and `local-deep-research`, we are re-architecting Med-STORM around a local RAG pipeline.

The new workflow will be:

1.  **Corpus Creation (Ingestion):** For a given research topic, perform an initial, one-time, broad search on primary data sources (like PubMed) to gather a comprehensive set of relevant documents.
2.  **Indexing:** Parse, chunk, and vectorize these documents using sentence-transformer models. Store the vectors and their corresponding metadata in a local vector database (e.g., Qdrant, Weaviate). This creates a searchable, semantic index of our research corpus.
3.  **Local-First Research (Retrieval & Synthesis):** The `StormEngine` will now query this local index. For each research question, it will perform a fast, efficient semantic similarity search to find the most relevant information *within the already-downloaded corpus*.
4.  **Synthesis:** The retrieved information is then passed to the LLM for synthesis, outlining, and report generation, as before.
5.  **Optional Fallback:** For information not found in the local corpus, the system can have a secondary, optional strategy to perform targeted live web searches.

![New Architecture Flow](https://www.med-storm.com/new_architecture.png) *Diagram to be created*

## 3. Key Technologies & Resources

We will leverage the following tools and projects as suggested:

| Component              | Technology/Project                                                                                      | Purpose                                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Vector Database**    | Qdrant / Weaviate                                                                                       | For efficient, local storage and semantic search of document embeddings. Enables the core RAG pipeline.                                |
| **Data Source (Primary)** | PubMed                                                                                                  | Initial source for building our medical research corpus.                                                                                 |
| **Parsing Utility**    | [`pubmed_parser`](https://github.com/titipata/pubmed_parser)                                            | To reliably parse and extract structured data (title, abstract, authors, etc.) from PubMed records during the ingestion phase.         |
| **Architectural Insight**| [`local-deep-research`](https://github.com/LearningCircuit/local-deep-research)                           | A reference for multi-source integration (arXiv, web search via SearXNG) and a well-implemented local RAG model.                 |
| **Advanced Concepts**  | [`cognee`](https://github.com/topoteretes/cognee)                                                         | A potential future resource for exploring advanced cognitive architectures to improve reasoning and synthesis quality.               |
| **Data Strategy**      | Med-PRM Datasource                                                                                      | A potential high-quality, curated dataset to use as a primary search tier before falling back to a dynamically built PubMed corpus.      |
| **Caching**            | Redis                                                                                                   | For caching LLM calls and other intermediate results to further speed up the process.                                                    |

## 4. Phased Implementation Plan

### Phase 1: Build the Core RAG Pipeline

-   **Step 1.1: Setup Docker Environment:** Define a `docker-compose.yml` to manage services like Qdrant and Redis.
-   **Step 1.2: Create the Ingestion Service:**
    -   Build a new module responsible for taking a topic, querying PubMed, and downloading a corpus of articles.
    -   Integrate `pubmed_parser` to process the results.
-   **Step 1.3: Implement Vectorization:**
    -   Add `sentence-transformers` to the project.
    -   Create a utility to convert article abstracts into vector embeddings.
    -   Store these embeddings in the Qdrant vector database.
-   **Step 1.4: Refactor the Search Connector:**
    -   Create a new `LocalCorpusConnector` (or adapt the existing `PubMedConnector`).
    -   This connector will take a research question, embed it, and query Qdrant to find the top-k most similar article chunks.
    -   It will replace the live API call logic.

### Phase 2: Enhance and Expand

-   **Step 2.1: Multi-Source Integration:** Following the `local-deep-research` pattern, add connectors for other sources like arXiv or a general web search (e.g., via Serper, which we already have).
-   **Step 2.2: Implement Hybrid Data Strategy:** Develop a tiered search system that first queries a high-quality, static dataset (if available, like Med-PRM's) before querying our dynamically built corpus.
-   **Step 2.3: Improve Synthesis:** Once the pipeline is robust, evaluate techniques from `cognee` or other advanced RAG methods to improve the final report's quality and coherence.

This strategic shift directly tackles our system's weaknesses and sets us on a path to build a truly powerful and innovative research tool. 