import os
from langchain_mistralai import MistralAIEmbeddings


LLM_TEMPERATURE = 0.1
MISTRAL_MODEL = "ministral-3b-2512"
EMBEDDING_MODEL = "mistral-embed"
PINECONE_INDEX_NAME = "pdf-structured-rag-mistral"
PINECONE_INDEX_DIMENSION = 1024
PINECONE_NAMESPACE = "default"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def get_embeddings() -> MistralAIEmbeddings:
    return MistralAIEmbeddings(model=EMBEDDING_MODEL, api_key=MISTRAL_API_KEY)