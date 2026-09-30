# Evaluation Summary — Free vs. Paid, and ReAct vs. Single-shot

*One-page version. Full detail: [`EVALUATION_REPORT.md`](./EVALUATION_REPORT.md). Date: 2026-09-26.*

## Bottom line
- **Retrieval is strong and identical (95.5% hit) across free & paid** — the chat model is *not*
  the lever for retrieval quality.
- **Paid model:** modestly better answers, but **~2× faster and far steadier** — that's the real win.
- **ReAct agent:** built and verified; expected to win specifically on **multi-hop / retrieval-miss**
  questions. Demonstrated on q22 (below). Full batch scores pending API quota.

## Free vs. Paid (✅ measured, 22 questions)

| Metric | Free | Paid | Note |
|---|---|---|---|
| Retrieval hit rate | 95.5% | 95.5% | identical (same embeddings) |
| Mean Reciprocal Rank | 0.818 | 0.818 | identical |
| LLM judge (1–5) | 4.00 | **4.18** | paid slightly better |
| ROUGE-L / BERTScore F1 | 0.433 / 0.833 | **0.499 / 0.871** | paid slightly better |
| **Avg latency** | 7.30s (spiky, up to 26s) | **3.51s (steady)** | **paid's biggest edge** |

**Read:** free is a credible zero-cost baseline (4/5 quality). Paid buys a small quality bump and
**halved, far more consistent latency** — worth it where responsiveness matters.

## How they work (in one look)

- **One-shot RAG:** `question → 1 search → 1 LLM call → answer`. Searches with the user's exact
  words; no way to react if that search is wrong.
- **ReAct agent:** `question → [Thought → Action(search) → Observation]× → finish`. Writes and
  **rewrites its own search queries**, can search several times, then answers. **The one decisive
  extra step is "the first result is weak → reformulate and search again"** — which one-shot lacks.

## ReAct vs. Single-shot

| | Single-shot (existing) | ReAct agent (new) |
|---|---|---|
| Retrievals per question | 1 (fixed) | multiple; can reformulate |
| Recovers from a bad first search | ❌ no | ✅ yes |
| Cost per question | 1 search / 1 LLM call | ~3 searches / ~4 LLM calls (measured on example) |
| Best for | simple factual lookups | multi-hop, "compare versions", retrieval misses |

**q22 case study — "If I quit, do I get paid for unused vacation days?"** (✅ mechanism measured)

| | Single-shot | ReAct mechanism |
|---|---|---|
| First search (user's raw wording) | wrong doc — **MISS** | same weak result |
| React | ❌ answers "not covered" → **judge 1/5** | ✅ reformulate → gold doc at **rank 1** |
| Answer available? | no | **yes — exact encashment clause** |

Root cause: user says "quit/vacation/paid," policy says "resignation/annual leave/encashment."
Single-shot is stuck with the user's phrasing; ReAct reformulates and recovers. *(Final ReAct
generation not run — blocked on API quota — but the bottleneck was retrieval, and paid already
scored 4/5 on q22 even with the wrong doc.)*

## Recommendations
1. **Paid for interactive** answering (faster, steadier); **free for batch/cost-sensitive**.
2. **Hybrid agent routing:** cheap single-shot for simple questions, ReAct for multi-hop/conflict —
   priced precisely via the built-in per-question call telemetry.
3. **To finish:** add a few $ of API credit (run cost < $1) or raise the free daily cap, then
   `python -m eval.evaluate --strategy single|react`. Also fix the open `generator.py:105` bug.

> ✅ measured = real harness numbers · ⚠️ not-yet-measured items are labelled as such; nothing is estimated as if measured.
