import argparse
import json
import random
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from pydantic import BaseModel, Field

from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE
from evals.judge import get_judge_llm

MIN_CHUNK_CHARS = 300
MAX_IDS_SCANNED = 5000


class QAPair(BaseModel):
    question: str = Field(description="A question a researcher might ask that this passage answers")
    reference_answer: str = Field(description="The correct answer, using only facts from the passage")


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write evaluation questions for a document question-answering system. Given a passage, write ONE "
            "specific question that the passage answers, and its reference answer using only the passage. "
            "The question must make sense on its own without seeing the passage: do not say 'the passage', "
            "'this section' or 'the author'. Prefer questions about substantive content, not formatting or citations.",
        ),
        ("human", "Passage:\n{passage}"),
    ]
)


def sample_chunks(n: int, seed: int):
    index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
    ids = []
    for batch in index.list(namespace=PINECONE_NAMESPACE):
        ids.extend(batch)
        if len(ids) >= MAX_IDS_SCANNED:
            break
    random.Random(seed).shuffle(ids)

    chunks = []
    for start in range(0, len(ids), 100):
        batch_ids = ids[start:start + 100]
        vectors = index.fetch(ids=batch_ids, namespace=PINECONE_NAMESPACE).vectors
        for vid in batch_ids:
            vec = vectors.get(vid)
            metadata = (vec.metadata if vec else None) or {}
            text = metadata.get("text", "")
            if len(text) >= MIN_CHUNK_CHARS:
                chunks.append((vid, text, metadata.get("source")))
                if len(chunks) == n:
                    return chunks
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of questions to generate")
    parser.add_argument("--out", default="evals/dataset.jsonl")
    args = parser.parse_args()

    chain = qa_prompt | get_judge_llm().with_structured_output(QAPair)
    chunks = sample_chunks(args.n, 42)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        for i, (chunk_id, text, source) in enumerate(chunks, start=1):
            qa = chain.invoke({"passage": text})
            row = {
                "question": qa.question,
                "reference_answer": qa.reference_answer,
                "source_chunk_id": chunk_id,
                "source": source,
            }
            f.write(json.dumps(row) + "\n")
            print(f"[{i}/{len(chunks)}] {qa.question}")
    print(f"\nWrote {len(chunks)} rows to {out}. Review them before running the eval.")


if __name__ == "__main__":
    main()
