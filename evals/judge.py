"""The LLM that grades the RAG pipeline (RAGAS metrics) and drafts the eval dataset."""
from functools import lru_cache

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_mistralai import ChatMistralAI

from app.config import JUDGE_MODEL, MISTRAL_API_KEY

# Stay under the judge model's per-minute request limit (30/min for ministral-14b on the free tier)
JUDGE_REQUESTS_PER_SECOND = 0.45


@lru_cache(maxsize=1)
def get_judge_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model=JUDGE_MODEL,
        temperature=0,
        max_retries=6,
        rate_limiter=InMemoryRateLimiter(requests_per_second=JUDGE_REQUESTS_PER_SECOND),
        api_key=MISTRAL_API_KEY,
    )
