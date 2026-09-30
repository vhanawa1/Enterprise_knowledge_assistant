"""
Evaluation harness.

Computes, per test question:
  - Retrieval recall@k: what fraction of the expected source document(s) were
    among the retrieved chunks. `retrieval_hit` is True only when ALL expected
    docs were retrieved (so single- and multi-hop questions score consistently).
  - MRR (mean reciprocal rank) of the first expected source in the results.
  - Answer keyword coverage: does the generated answer contain the expected
    key facts (a cheap proxy that doesn't need a reference answer at all).
  - ROUGE-1/2/L, BERTScore, and an LLM-as-judge score, each comparing the
    generated answer against the gold `expected_answer` in the testset.
  - Confidence label returned, end-to-end latency, and -- for the agent --
    the cost of the strategy (LLM calls and searches per question).

Two answering strategies can be evaluated with identical metrics:
  --strategy single  (default)  single retrieval + one LLM call (rag.generator)
  --strategy react              ReAct agent, multi-step retrieval (rag.react_agent)

Results are written to eval_results_{mode}.json (single) or
eval_results_react_{mode}.json (react), which the eval dashboard visualizes.

Run:
    python -m eval.evaluate                 # single-shot baseline
    python -m eval.evaluate --strategy react
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from rag.vector_store import VectorStore
from rag.retriever import retrieve_and_rerank
from rag.generator import generate_answer
from rag.react_agent import answer_with_react
from eval.metrics import compute_rouge, compute_bertscore_batch, llm_judge


def load_testset(path: str = config.EVAL_TESTSET_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def results_path(strategy: str) -> str:
    if strategy == "react":
        return str(Path(config.EVAL_RESULTS_PATH).with_name(f"eval_results_react_{config.RAG_MODE}.json"))
    return config.EVAL_RESULTS_PATH


def _gold_sources(item: dict) -> list[str] | None:
    """Normalize the gold source(s) into a list, or None for refusal cases.
    Supports both `expected_source` (str|null) and `expected_sources` (list)."""
    if item.get("expected_sources"):
        return list(item["expected_sources"])
    src = item.get("expected_source")
    return [src] if src else None


def _run_strategy(question: str, store: VectorStore, strategy: str) -> dict:
    """Return a generate_answer-shaped result dict for the chosen strategy.
    Both paths populate `chunks_used`, so retrieval metrics read from one place.
    Single-shot has a fixed cost of 1 search + 1 LLM call; the agent reports
    its own counts."""
    if strategy == "react":
        return answer_with_react(question, store)
    chunks = retrieve_and_rerank(question, store)
    result = generate_answer(question, chunks)
    result.setdefault("num_llm_calls", 1)
    result.setdefault("num_searches", 1)
    result.setdefault("steps", 1)
    return result


def evaluate_question(item: dict, store: VectorStore, strategy: str) -> dict:
    t0 = time.time()
    result = _run_strategy(item["question"], store, strategy)
    latency = time.time() - t0

    retrieved_sources = [c["metadata"].get("source") for c in result.get("chunks_used", [])]
    gold = _gold_sources(item)

    if gold is None:
        # negative/refusal case: correct behavior is judged via answer coverage,
        # not retrieval. Absence of a doc isn't itself a failure.
        retrieval_hit = True
        recall = None
        rank = None
    else:
        found = [g for g in gold if g in retrieved_sources]
        recall = len(found) / len(gold)
        retrieval_hit = (recall == 1.0)          # all required docs retrieved
        ranks = [retrieved_sources.index(g) + 1 for g in found]
        rank = min(ranks) if ranks else None      # rank of first gold doc

    mrr = (1 / rank) if rank else 0.0

    answer_lower = result["answer"].lower()
    keywords = item.get("expected_answer_contains", [])
    matched_keywords = [kw for kw in keywords if kw.lower() in answer_lower]
    keyword_coverage = len(matched_keywords) / len(keywords) if keywords else None

    reference_answer = item.get("expected_answer")
    rouge = compute_rouge(reference_answer, result["answer"]) if reference_answer else None
    judge = llm_judge(item["question"], reference_answer, result["answer"]) if reference_answer else None

    return {
        "id": item["id"],
        "question": item["question"],
        "category": item.get("category", "standard"),
        "gold_sources": gold,
        "retrieved_sources": retrieved_sources,
        "retrieval_hit": retrieval_hit,
        "retrieval_recall": recall,
        "mrr": mrr,
        "answer": result["answer"],
        "reference_answer": reference_answer,
        "confidence": result["confidence"],
        "confidence_score": result["confidence_score"],
        "keyword_coverage": keyword_coverage,
        "matched_keywords": matched_keywords,
        "expected_keywords": keywords,
        "rouge": rouge,
        "llm_judge_score": judge["score"] if judge else None,
        "llm_judge_reasoning": judge["reasoning"] if judge else None,
        "latency_sec": round(latency, 2),
        "num_llm_calls": result.get("num_llm_calls"),
        "num_searches": result.get("num_searches"),
    }


def run_evaluation(strategy: str = "single", testset_path: str = config.EVAL_TESTSET_PATH,
                   out_path: str | None = None) -> dict:
    out_path = out_path or results_path(strategy)
    store = VectorStore()
    if store.count() == 0:
        raise RuntimeError("Vector store is empty. Run ingestion first: python -m ingestion.ingest")

    testset = load_testset(testset_path)
    print(f"[INFO] Evaluating {len(testset)} questions with strategy='{strategy}' "
          f"(mode={config.RAG_MODE}, model={config.OPENROUTER_CHAT_MODEL if config.RAG_MODE=='free' else config.CHAT_MODEL})")
    results = []
    for i, item in enumerate(testset, 1):
        r = evaluate_question(item, store, strategy)
        results.append(r)
        print(f"  [{i}/{len(testset)}] {item['id']}: hit={r['retrieval_hit']} "
              f"conf={r['confidence']} calls={r['num_llm_calls']} {r['latency_sec']}s")

    # BERTScore loads a model, so compute it once for the whole batch.
    scoreable = [(i, r) for i, r in enumerate(results) if r["reference_answer"]]
    if scoreable:
        candidates = [r["answer"] for _, r in scoreable]
        references = [r["reference_answer"] for _, r in scoreable]
        bertscores = compute_bertscore_batch(candidates, references)
        for (i, _), bs in zip(scoreable, bertscores):
            results[i]["bertscore"] = bs
    for r in results:
        r.setdefault("bertscore", None)

    n = len(results)

    def _avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    hit_rate = sum(r["retrieval_hit"] for r in results) / n
    avg_recall = _avg([r["retrieval_recall"] for r in results])
    avg_mrr = sum(r["mrr"] for r in results) / n
    avg_keyword_coverage = _avg([r["keyword_coverage"] for r in results])
    avg_latency = sum(r["latency_sec"] for r in results) / n
    avg_llm_calls = _avg([r["num_llm_calls"] for r in results])
    avg_searches = _avg([r["num_searches"] for r in results])
    confidence_dist = {}
    for r in results:
        confidence_dist[r["confidence"]] = confidence_dist.get(r["confidence"], 0) + 1

    avg_rouge_l = _avg([r["rouge"]["rougeL"] for r in results if r["rouge"]])
    avg_bertscore_f1 = _avg([r["bertscore"]["f1"] for r in results if r["bertscore"]])
    avg_llm_judge_score = _avg([r["llm_judge_score"] for r in results])

    def _rnd(v, d=3):
        return round(v, d) if v is not None else None

    summary = {
        "strategy": strategy,
        "num_questions": n,
        "retrieval_hit_rate": _rnd(hit_rate),
        "avg_retrieval_recall": _rnd(avg_recall),
        "mean_reciprocal_rank": _rnd(avg_mrr),
        "avg_answer_keyword_coverage": _rnd(avg_keyword_coverage),
        "avg_rouge_l": _rnd(avg_rouge_l),
        "avg_bertscore_f1": _rnd(avg_bertscore_f1),
        "avg_llm_judge_score": _rnd(avg_llm_judge_score, 2),
        "avg_latency_sec": _rnd(avg_latency, 2),
        "avg_llm_calls_per_q": _rnd(avg_llm_calls, 2),
        "avg_searches_per_q": _rnd(avg_searches, 2),
        "confidence_distribution": confidence_dist,
    }

    output = {"summary": summary, "results": results}
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\n[DONE] Full results written to {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a RAG answering strategy.")
    parser.add_argument("--strategy", choices=["single", "react"], default="single",
                        help="single = one retrieval + one LLM call; react = multi-step agent")
    args = parser.parse_args()
    run_evaluation(strategy=args.strategy)
