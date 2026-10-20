from langchain_mistralai import MistralAIEmbeddings

from app.config import EMBEDDING_MODEL, MISTRAL_API_KEY


def get_embeddings() -> MistralAIEmbeddings:
    return MistralAIEmbeddings(model=EMBEDDING_MODEL, api_key=MISTRAL_API_KEY)