import os
from dotenv import load_dotenv

load_dotenv()

# Embedding model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
# Alternative: "text-embedding-ada-002" for OpenAI

# OpenAI configuration (if using OpenAI embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Chunking configuration
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Vector index configuration
VECTOR_INDEX_NAME = "chunk_embeddings"
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))  # all-mpnet-base-v2 outputs 768-dim vectors

# NER configuration
NER_MODEL = os.getenv("NER_MODEL", "dslim/bert-base-NER")
USE_SPACY_FALLBACK = os.getenv("USE_SPACY_FALLBACK", "true").lower() == "true"
