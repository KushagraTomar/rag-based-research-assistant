"""Evaluate the RAG pipeline with RAGAS.

    python -m evals.run --dataset evals/dataset.jsonl [--limit 10]

Dataset: JSONL, one object per line: {"question": "...", "reference_answer": "..."}
"""
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

# RAGAS 0.4 marks its evaluate()-compatible metrics as deprecated in favour of
# ragas.metrics.collections, which can't run through evaluate() yet
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextRelevance,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from app.config import JUDGE_MODEL, MISTRAL_MODEL, RETRIEVAL_K
from app.rag.rag import rag_pipeline
from app.retrieval.embedding import get_embeddings
from evals.judge import get_judge_llm

METRICS = {
    "Retrieval": [LLMContextPrecisionWithReference(), LLMContextRecall()],
    # strictness=1: with several prompts per call, langchain-mistralai crashes summing Mistral's nested token usage
    "Generation": [Faithfulness(), ResponseRelevancy(strictness=1), ContextRelevance()],
}


def collect_samples(rows: list) -> list:
    """Run each question through the RAG pipeline and record what RAGAS needs to score it."""
    samples = []
    for i, row in enumerate(rows, start=1):
        state = rag_pipeline.run(row["question"])
        samples.append({
            "user_input": row["question"],
            "retrieved_contexts": [doc.page_content for doc in state.get("documents", [])],
            "response": state["answer"],
            "reference": row["reference_answer"],
        })
        print(f"[{i}/{len(rows)}] {row['question'][:80]}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/dataset.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="evals/results")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    rows = rows[: args.limit] if args.limit else rows

    print("Running RAG pipeline...")
    samples = collect_samples(rows)

    print("\nScoring with RAGAS...")
    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=[metric for group in METRICS.values() for metric in group],
        llm=LangchainLLMWrapper(get_judge_llm()),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
        # Few workers + patient retries: the judge model has a low per-minute rate limit
        run_config=RunConfig(max_workers=4, max_retries=10, max_wait=60, timeout=300),
    )

    df = result.to_pandas()
    summary = {
        metric.name: (None if df[metric.name].isna().all() else round(float(df[metric.name].mean()), 3))
        for group in METRICS.values() for metric in group
    }

    run_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "results.csv", index=False)
    config = {"generator": MISTRAL_MODEL, "judge": JUDGE_MODEL, "k": RETRIEVAL_K,
              "dataset": args.dataset, "rows": len(samples)}
    (run_dir / "summary.json").write_text(json.dumps({"config": config, "metrics": summary}, indent=2))

    print(f"\n{'Metric':<40}{'Component':<12}Score")
    print("-" * 58)
    for component, metrics in METRICS.items():
        for metric in metrics:
            value = summary[metric.name]
            failed = int(df[metric.name].isna().sum())
            note = f"  ({failed} rows failed)" if failed else ""
            print(f"{metric.name:<40}{component:<12}{'n/a' if value is None else f'{value:.3f}'}{note}")
    print(f"\nSaved to {run_dir}/")


if __name__ == "__main__":
    main()
