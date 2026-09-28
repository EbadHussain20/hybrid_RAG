"""Evaluation harness.

Run with: python -m evaluation.run

Produces:
  evaluation/results/results.json
  evaluation/results/report.md

Retrieval metrics (Recall@K, Precision@K, MRR, Hit Rate) are computed by
matching each retrieved chunk's source_path against the question's
`expected_sources` list (filename stems) - a chunk "hits" if its source file
stem is one of the expected sources. This is computed independently for
Dense-only, Sparse-only, and Hybrid retrieval to demonstrate hybrid's effect,
which is the whole point of this comparison per the project spec.

Citation correctness/completeness and groundedness are measured by actually
running the full RagPipeline (using the Mock LLM provider, since no
network path to Gemini exists in this environment - see README) for every
non-"NO_ANSWER" question and recording real citation-validation/grounding
output, not fabricated numbers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.core.config import settings
from app.core.indexer import load_retrievers
from app.core.rag_pipeline import RagPipeline
from app.retrieval.hybrid import HybridRetriever

EVAL_DATASET_PATH = Path(__file__).parent / "datasets" / "eval_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"
K_VALUES = (1, 3, 5)


def _matches_expected(source_path: str, expected_sources: list[str]) -> bool:
    stem = Path(source_path).stem
    return any(exp == stem for exp in expected_sources)


def _retrieval_metrics_for_mode(records: list[dict], get_ranked_paths) -> dict:
    """get_ranked_paths(question, expected_sources) -> list[str] of source_paths in rank order."""
    metrics = {f"recall@{k}": [] for k in K_VALUES}
    metrics.update({f"precision@{k}": [] for k in K_VALUES})
    mrr_scores = []
    hit_rate_hits = 0
    eval_count = 0

    for rec in records:
        expected = rec["expected_sources"]
        if not expected:
            continue  # no-answer questions have no retrieval ground truth
        eval_count += 1
        ranked_paths = get_ranked_paths(rec["question"])

        hit_ranks = [i + 1 for i, p in enumerate(ranked_paths) if _matches_expected(p, expected)]
        hit_rate_hits += 1 if hit_ranks else 0
        mrr_scores.append(1.0 / hit_ranks[0] if hit_ranks else 0.0)

        for k in K_VALUES:
            top_k_paths = ranked_paths[:k]
            hits_at_k = sum(1 for p in top_k_paths if _matches_expected(p, expected))
            metrics[f"recall@{k}"].append(min(1.0, hits_at_k / len(set(expected))))
            metrics[f"precision@{k}"].append(hits_at_k / max(1, len(top_k_paths)))

    result = {k: round(sum(v) / len(v), 3) if v else 0.0 for k, v in metrics.items()}
    result["mrr"] = round(sum(mrr_scores) / len(mrr_scores), 3) if mrr_scores else 0.0
    result["hit_rate"] = round(hit_rate_hits / eval_count, 3) if eval_count else 0.0
    result["num_questions_evaluated"] = eval_count
    return result


def run_retrieval_comparison(dense, sparse, records: list[dict]) -> dict:
    hybrid = HybridRetriever(dense, sparse)

    def dense_paths(q: str) -> list[str]:
        return [c.source_path for c, _, _ in dense.search(q, top_k=max(K_VALUES))]

    def sparse_paths(q: str) -> list[str]:
        return [c.source_path for c, _, _, _ in sparse.search(q, top_k=max(K_VALUES))]

    def hybrid_paths(q: str) -> list[str]:
        return [c.chunk.source_path for c in hybrid.retrieve(q, top_k=max(K_VALUES))]

    return {
        "dense_only": _retrieval_metrics_for_mode(records, dense_paths),
        "sparse_only": _retrieval_metrics_for_mode(records, sparse_paths),
        "hybrid_rrf": _retrieval_metrics_for_mode(records, hybrid_paths),
    }


def run_generation_eval(pipeline: RagPipeline, records: list[dict]) -> dict:
    per_question = []
    citation_valid_count = 0
    grounded_count = 0
    correct_no_answer = 0
    total_no_answer = 0
    total_evaluated = 0

    for rec in records:
        total_evaluated += 1
        resp = pipeline.answer_question(rec["question"])
        is_no_answer_expected = rec["expected_answer"] == "NO_ANSWER"
        produced_no_answer = "couldn't find enough information" in resp.answer.lower()

        if is_no_answer_expected:
            total_no_answer += 1
            if produced_no_answer:
                correct_no_answer += 1

        if resp.citation_validation.valid:
            citation_valid_count += 1
        if resp.grounding.supported:
            grounded_count += 1

        per_question.append({
            "id": rec["id"],
            "category": rec["category"],
            "question": rec["question"],
            "expected_no_answer": is_no_answer_expected,
            "produced_no_answer": produced_no_answer,
            "citations_valid": resp.citation_validation.valid,
            "num_citations": resp.citation_validation.citations_checked,
            "grounded": resp.grounding.supported,
            "total_latency_ms": round(resp.latency.total_ms, 2),
        })

    return {
        "citation_validity_rate": round(citation_valid_count / total_evaluated, 3),
        "groundedness_rate": round(grounded_count / total_evaluated, 3),
        "no_answer_accuracy": round(correct_no_answer / total_no_answer, 3) if total_no_answer else None,
        "num_questions": total_evaluated,
        "per_question": per_question,
    }


def _write_report_md(retrieval_comparison: dict, generation_eval: dict, out_path: Path) -> None:
    lines = ["# Evaluation Report", "", f"Dataset: `{EVAL_DATASET_PATH.name}` "
             f"({sum(1 for _ in json.loads(EVAL_DATASET_PATH.read_text()))} questions)", ""]

    lines.append("## Retrieval: Dense vs Sparse vs Hybrid")
    lines.append("")
    lines.append("| Metric | Dense Only | Sparse (BM25) Only | Hybrid (RRF) |")
    lines.append("|---|---|---|---|")
    for metric in ["recall@1", "recall@3", "recall@5", "precision@1", "precision@3",
                   "precision@5", "mrr", "hit_rate"]:
        d = retrieval_comparison["dense_only"][metric]
        s = retrieval_comparison["sparse_only"][metric]
        h = retrieval_comparison["hybrid_rrf"][metric]
        lines.append(f"| {metric} | {d} | {s} | {h} |")
    lines.append("")
    lines.append(f"(Evaluated over {retrieval_comparison['hybrid_rrf']['num_questions_evaluated']} "
                 "questions that have ground-truth expected sources; no-answer questions are excluded "
                 "from retrieval metrics since they have no ground truth to retrieve.)")
    lines.append("")

    lines.append("## Generation: Citation & Grounding Quality")
    lines.append("")
    lines.append(f"- Citation validity rate: **{generation_eval['citation_validity_rate']:.0%}** "
                 "(fraction of answers whose citation markers all map to real retrieved chunks)")
    lines.append(f"- Groundedness rate: **{generation_eval['groundedness_rate']:.0%}** "
                 "(answers whose vocabulary is substantially supported by retrieved context)")
    if generation_eval["no_answer_accuracy"] is not None:
        lines.append(f"- No-answer accuracy: **{generation_eval['no_answer_accuracy']:.0%}** "
                     "(correctly declined to answer when no evidence exists)")
    lines.append(f"- Questions evaluated: {generation_eval['num_questions']}")
    lines.append("")

    lines.append("## Per-Question Results")
    lines.append("")
    lines.append("| ID | Category | Citations Valid | Grounded | No-Answer (expected/produced) | Latency (ms) |")
    lines.append("|---|---|---|---|---|---|")
    for q in generation_eval["per_question"]:
        lines.append(f"| {q['id']} | {q['category']} | {q['citations_valid']} | {q['grounded']} | "
                     f"{q['expected_no_answer']}/{q['produced_no_answer']} | {q['total_latency_ms']} |")

    out_path.write_text("\n".join(lines))


def main():
    print(f"Loading index and building pipeline (embedding_provider={settings.EMBEDDING_PROVIDER}, "
          f"fusion={settings.FUSION_METHOD}, llm_provider={settings.LLM_PROVIDER})...")
    dense, sparse = load_retrievers()
    if not dense.chunks:
        raise SystemExit("Index is empty. Run `python scripts/ingest.py` and `python scripts/index.py` first.")

    pipeline = RagPipeline(dense, sparse)
    records = json.loads(EVAL_DATASET_PATH.read_text())

    print(f"Running retrieval comparison over {len(records)} questions (dense vs sparse vs hybrid)...")
    t0 = time.time()
    retrieval_comparison = run_retrieval_comparison(dense, sparse, records)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Running full generation pipeline for citation/groundedness evaluation...")
    t0 = time.time()
    generation_eval = run_generation_eval(pipeline, records)
    print(f"  done in {time.time() - t0:.1f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_payload = {
        "num_questions": len(records),
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "fusion_method": settings.FUSION_METHOD,
        "llm_provider_used": generation_eval["per_question"][0] if generation_eval["per_question"] else None,
        "retrieval_comparison": retrieval_comparison,
        "generation_eval": {k: v for k, v in generation_eval.items() if k != "per_question"},
        "per_question": generation_eval["per_question"],
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(results_payload, indent=2))
    _write_report_md(retrieval_comparison, generation_eval, RESULTS_DIR / "report.md")

    print("\n=== Retrieval Comparison (Recall@5) ===")
    print(f"  Dense only:  {retrieval_comparison['dense_only']['recall@5']}")
    print(f"  Sparse only: {retrieval_comparison['sparse_only']['recall@5']}")
    print(f"  Hybrid RRF:  {retrieval_comparison['hybrid_rrf']['recall@5']}")
    print(f"\nCitation validity rate: {generation_eval['citation_validity_rate']:.0%}")
    print(f"Groundedness rate: {generation_eval['groundedness_rate']:.0%}")
    if generation_eval["no_answer_accuracy"] is not None:
        print(f"No-answer accuracy: {generation_eval['no_answer_accuracy']:.0%}")
    print(f"\nWrote {RESULTS_DIR / 'results.json'} and {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
