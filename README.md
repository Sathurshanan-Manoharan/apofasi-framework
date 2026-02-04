# Apofasi: Temporal-Aware Legal RAG Framework

Apofasi is a unified retrieval-augmented generation framework designed for evolving mixed legal systems, specifically tailored for the Sri Lankan legal context. It handles statutory evolution (amendments, repeals) and links case law to applicable statutes, enabling research that is sensitive to the temporal validity of laws.

## Overview

Key capabilities include:
- **Temporal Validity Tracking**: Accurate handling of law as it existed at any specific point in time, managing repeals and amendments.
- **Statute-Case Linkage**: Automatically linking case judgments to the specific statutory provisions they cite.
- **Hybrid Search**: Combining vector semantic search with graph-based structural queries.

## Architecture

The system follows a linear pipeline:
1. **Ingestion**: Processing PDF documents (Statutes and Cases).
2. **Extraction**: Identifying temporal metadata, citations, and sections using NER and regex.
3. **Modeling**: Constructing a knowledge graph in Neo4j with vector embeddings.
4. **Retrieval**: Hybrid search enabling temporal filtering and multi-hop reasoning.

### Technology Stack
- **Database**: Neo4j (Graph + Vector Index)
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **NER**: BERT-based models + Regex fallbacks
- **Framework**: Python 3.9+, LangChain

## Setup

### Prerequisites
1. **Neo4j 5.11+**: Required for vector index support.
2. **Python 3.9+**

### Installation

```bash
cd "apofasi backend"
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configuration

1. Copy the template configuration:
   ```bash
   cp .env.template .env
   ```

2. Update `.env` with your Neo4j credentials and API keys.

3. Initialize the database schema:
   ```bash
   python modeling_3/graph_schema/init_schema.py
   ```

## Usage

### Full Pipeline Execution
To process the complete statutes dataset:
```bash
python main.py --full --source statutes
```

This ingests raw PDFs, extracts metadata, builds the graph, and establishes temporal relationships.

### Component-Level Commands

**Ingestion Only:**
```bash
python ingestion_1/run_ingestion.py --source statutes
```

**Schema Initialization:**
```bash
python main.py --schema-only
```

**Testing:**
Run vertical slice tests to verify temporal correctness:
```bash
python evaluation/run_vertical_slice_tests.py
```

## Project Structure

- `data/`: Raw and processed document storage.
- `ingestion_1/`: PDF text extraction and chunking.
- `extraction_2/`: Metadata and citation extraction logic.
- `modeling_3/`: Graph construction and vector operations.
- `retrieval_4/`: Search logic including temporal filtering.
- `evaluation/`: automated tests for system validation.

## License

[License Information]