import os

LLM_TEMPERATURE = 0.1
MISTRAL_MODEL = "ministral-3b-2512"
JUDGE_MODEL = "ministral-14b-2512"
EMBEDDING_MODEL = "mistral-embed"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 5

PINECONE_INDEX_NAME = "pdf-structured-rag-mistral"
PINECONE_INDEX_DIMENSION = 1024
PINECONE_NAMESPACE = "default"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")