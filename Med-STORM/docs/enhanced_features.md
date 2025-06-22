# Enhanced Features Documentation

This document provides detailed information about the enhanced features implemented in the Med-STORM system.

## Table of Contents

1. [Evidence Tiers](#evidence-tiers)
2. [Streaming Search](#streaming-search)
3. [Personalized Medicine](#personalized-medicine)
4. [Deduplication](#deduplication)
5. [Comparison Tables](#comparison-tables)
6. [Executive Summaries](#executive-summaries)
7. [Configuration](#configuration)

## Evidence Tiers

Med-STORM now classifies evidence into three distinct tiers based on source reliability and study quality:

### Tier 1: Highest Confidence (95%)
- **Sources**: PubMed, ClinicalTrials.gov
- **Description**: Peer-reviewed clinical trials, systematic reviews, and meta-analyses
- **Use Case**: Primary evidence for clinical decision making

### Tier 2: High Confidence (80%)
- **Sources**: Google Scholar, Clinical Guidelines, Government Health Organizations (WHO, CDC, NICE)
- **Description**: Guidelines from professional organizations and high-quality observational studies
- **Use Case**: Supporting evidence and clinical guidelines

### Tier 3: Standard Confidence (65%)
- **Sources**: Preprints, Conference Abstracts, General Medical Websites
- **Description**: Preliminary findings and general medical information
- **Use Case**: Background information and emerging research

## Streaming Search

The enhanced search system now supports streaming of results for improved performance with large result sets.

### Key Features:
- Progressive loading of search results
- Configurable chunk sizes
- Timeout handling
- Memory efficiency

### Example Usage:

```python
async for chunk in connector.search_stream(
    query="diabetes treatment",
    max_results=100,
    min_publication_year=2020
):
    process_chunk(chunk)
```

## Personalized Medicine

The system now generates personalized treatment recommendations based on patient-specific factors.

### Supported Patient Factors:
- **Demographic**: Age, gender, ethnicity
- **Clinical**: Diagnoses, lab results, vitals
- **Genetic**: Genomic markers, family history
- **Lifestyle**: Diet, exercise, smoking status
- **Comorbidities**: Existing medical conditions
- **Medications**: Current medications
- **Allergies**: Known allergies and adverse reactions
- **Social**: Living situation, support system

### Example Usage:

```python
recommendations = await engine.generate_personalized_recommendations(
    corpus=evidence_corpus,
    patient_factors=[
        {"name": "age", "value": "65", "type": "demographic"},
        {"name": "diabetes", "value": "Type 2", "type": "comorbidity"}
    ],
    condition="Type 2 Diabetes"
)
```

## Deduplication

Advanced deduplication ensures that only unique evidence is presented.

### Methods:
1. **Exact Matching**: Identifies identical content
2. **SimHash**: Detects near-duplicate content
3. **TF-IDF + Cosine Similarity**: Identifies semantically similar content
4. **Hybrid**: Combines multiple methods for best results

### Configuration:
```python
# In settings.py
DEDUPLICATION_ENABLED = True
DEDUPLICATION_METHOD = "hybrid"  # exact, simhash, tfidf, hybrid
SIMILARITY_THRESHOLD = 0.85
```

## Comparison Tables

Generate structured tables to compare evidence sources.

### Table Types:
1. **Evidence Summary**: Key information across studies
2. **Risk of Bias**: Study quality assessment
3. **Outcome Comparison**: Treatment outcomes across studies

### Example:
```python
table = generate_evidence_summary_table(
    sources=evidence_sources,
    columns=["study_type", "sample_size", "outcome", "confidence"],
    sort_by="publication_date",
    descending=True
)
```

## Executive Summaries

Automatically generate comprehensive summaries of research findings.

### Sections:
1. **Overview**: Brief summary of key findings
2. **Key Results**: Detailed results with confidence levels
3. **Methodology**: Search strategy and inclusion criteria
4. **Clinical Implications**: Practical applications
5. **Limitations**: Study weaknesses and biases
6. **Recommendations**: Evidence-based suggestions

### Configuration:
```python
# In settings.py
SUMMARY_TARGET_LENGTH = 1000  # words
SUMMARY_INCLUDE_TABLES = True
SUMMARY_CONFIDENCE_THRESHOLD = 0.7
```

## Configuration

### Key Settings:

```python
# Search Settings
MAX_SEARCH_RESULTS = 20
MIN_PUBLICATION_YEAR = 2018
SORT_BY_DATE = True

# Performance
MAX_CONCURRENT_SEARCHES = 5
BATCH_SIZE_QUESTIONS = 5
BATCH_SIZE_SOURCES = 20

# Output
MAX_TABLE_ROWS = 50
MAX_TABLE_COLUMNS = 8
OUTPUT_DIR = "output"
```

### Environment Variables:

```bash
# Required
OPENAI_API_KEY=your_openai_key
PUBMED_EMAIL=your_email@example.com

# Optional
PUBMED_API_KEY=your_pubmed_key
LOG_LEVEL=INFO
CACHE_ENABLED=true
```

## Best Practices

1. **For Clinical Use**:
   - Prefer Tier 1 evidence for critical decisions
   - Always review the full text of important studies
   - Consider patient-specific factors when applying recommendations

2. **For Research**:
   - Use the streaming API for large literature reviews
   - Leverage the deduplication features to avoid redundant analysis
   - Customize the confidence thresholds based on your needs

3. **For Development**:
   - Monitor API usage and performance metrics
   - Adjust batch sizes based on available resources
   - Implement proper error handling for production use
