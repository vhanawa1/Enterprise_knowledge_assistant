# Enterprise Knowledge Assistant — Evaluation Report

**Prepared for:** team review
**Scope:** (A) Free vs. Paid model comparison, (B) ReAct agent vs. the existing single-shot RAG agent
**Date:** 2026-09-26

> **How to read this report.** Two very different kinds of evidence appear below, and they are
> labelled throughout:
> - **✅ Measured** — numbers produced by our evaluation harness on the 22-question test set. These are real.
> - **🔷 Design / Expected** — architectural analysis plus one fully-worked, verified example. The
>   ReAct agent's *aggregate* metrics have **not** been measured yet (see [Status](#status--what-is-left)),
>   so its section is analysis + a single real example, not a batch score. Nothing here is fabricated.

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Does the paid model beat the free model? | **Yes, modestly, on answer quality — and it is ~2× faster and far more consistent.** Retrieval quality is identical (same embeddings), so the gap is purely in generation. |
| Is retrieval the bottleneck or generation? | **Retrieval is strong (95.5% hit rate) and identical across free/paid.** The differences are in how well each model *writes the answer* from the same retrieved context, and in latency. |
| Does the ReAct agent help? | **Expected to help specifically on multi-hop / retrieval-miss questions** — the exact cases where the single-shot agent measurably fails today (e.g. q22). Verified on one worked example; full batch comparison is built and ready but pending API quota. |
| What does ReAct cost? | **~4× more LLM calls and multiple retrievals per question** (measured: 4 LLM calls / 3 searches on the worked example vs. a fixed 1 / 1 for single-shot). It is a quality-for-cost trade. |

---

## 2. Setup & methodology

- **Corpus / retrieval (held constant across every run):** ChromaDB vector store, local
  `all-MiniLM-L6-v2` embeddings, hybrid dense + BM25 fusion, cross-encoder reranking
  (`ms-marco-MiniLM-L-6-v2`). Keeping retrieval identical means any free-vs-paid or
  single-vs-ReAct difference is attributable to the **model / orchestration**, not to retrieval.
- **Test set:** 22 graded questions with gold source documents and reference answers
  (later extended to 28 with 6 purpose-built multi-hop questions — see Part B).
- **Metrics:**
  - *Retrieval:* hit rate (all gold docs retrieved), Mean Reciprocal Rank (MRR).
  - *Answer quality:* keyword coverage, ROUGE-L (lexical overlap), BERTScore F1 (semantic
    overlap), and an LLM-as-judge correctness score (1–5).
  - *Operational:* end-to-end latency; and for the agent, **cost telemetry** (LLM calls and
    searches per question).
- **Free tier** = chat via an OpenRouter free model + local embeddings (no API billing).
  **Paid tier** = a hosted commercial chat model. (The result files do not record the exact
  model strings; they are compared by *tier*.)

---

## 3. Part A — Free vs. Paid (✅ Measured, 22 questions)

### 3.1 Headline numbers

| Metric | Free | Paid | Winner |
|---|---|---|---|
| Retrieval hit rate | **95.5%** | **95.5%** | tie (same retrieval) |
| Mean Reciprocal Rank | **0.818** | **0.818** | tie (same retrieval) |
| Answer keyword coverage | 62.9% | **67.2%** | paid |
| ROUGE-L | 0.433 | **0.499** | paid |
| BERTScore F1 | 0.833 | **0.871** | paid |
| LLM judge (1–5) | 4.00 | **4.18** | paid |
| Avg latency | 7.30s | **3.51s** | **paid (2.1× faster)** |
| Confidence dist. | 22× High | 22× High | tie |

### 3.2 What the numbers say

1. **Retrieval is not the differentiator.** Hit rate and MRR are *identical* to three decimals
   because both tiers use the same local embeddings and reranker. If we want to improve
   retrieval, changing the chat model is the wrong lever — that work lives in chunking/embeddings/reranking.

2. **Paid wins on generation quality, but the margin is modest.** Across ROUGE-L (+0.066),
   BERTScore F1 (+0.038) and LLM-judge (+0.18 / 5), the paid model produces answers that align
   better with the reference text. The free model is **not far behind** and clears a 4/5 average —
   good enough for many internal use cases.

3. **The biggest, most decision-relevant gap is latency and consistency.** Paid averaged **3.51s**
   vs the free tier's **7.30s**, and the free tier was *spiky* — several questions took 11–26s
   (q1 26s, q7 19.5s, q9 14.2s, q15 14.2s, q17 11.6s), whereas paid stayed ~1.5–2.5s almost
   everywhere. For an interactive assistant, that tail latency is felt by users more than a 0.18
   judge-point difference.

### 3.3 Concrete examples (from the measured runs)

- **q22 — "If I quit, do I get paid for unused vacation days?"** The **only retrieval miss** for
  *both* tiers (hit = False). Free scored judge **1/5**; paid scored **4/5** on the *same* missed
  context — i.e. the paid model degraded more gracefully (hedged / reasoned) when the evidence
  wasn't retrieved. **This is the signature case for the ReAct agent** (Part B): a single retrieval
  pass missed the document, and neither single-shot agent could recover because neither can search
  again.
- **q15 — free keyword coverage 0.0 but judge 5/5.** A reminder that keyword coverage is a crude
  proxy; the free answer was judged fully correct while using different wording. Trust the
  judge/BERTScore over raw keyword coverage for quality claims.
- **q7 & q1** — the two largest paid-over-free judge gains (+1 and +2), both also cases where the
  free model was slow (19.5s, 26s). Paid was both better *and* faster here.

**Takeaway for the team:** Free is a credible, zero-cost baseline (4/5 quality, identical
retrieval). Paid buys a modest quality bump and — more importantly — **halved, far steadier
latency**. Choose paid where interactive responsiveness matters; free is fine for batch/internal
or cost-sensitive use.

---

## 4. Part B — ReAct agent vs. single-shot agent

### 4.1 The architectural difference (🔷 Design)

| | **Single-shot (existing)** | **ReAct agent (new)** |
|---|---|---|
| Retrieval | Exactly **one** retrieval, fixed top-k | **Multiple** retrievals; can reformulate the query |
| Reasoning | One LLM call: context → answer | Thought → Action → Observation loop (up to 5 steps) |
| Multi-doc questions | Limited to what one top-k pass returns | Can gather each sub-topic in separate searches |
| Recovery from a bad first search | **None** — answers from whatever came back | Can notice a weak result and **search again** |
| Cost per question | Fixed: **1 search, 1 LLM call** | Variable: **several searches + LLM calls** |
| Failure mode | Confidently answers from wrong/partial context | Higher latency & token cost |

The single-shot agent's structural weakness is visible in the measured data: **q22 failed because
one retrieval pass missed the document and there was no second chance.** The ReAct agent is
designed precisely to convert that class of failure into a success by reformulating and searching
again.

### 4.2 Verified worked example (✅ Real, single question)

We ran the ReAct agent end-to-end on a deliberately multi-hop question:

> *"What is the password rotation requirement and how did it change between policy versions?"*

**Result (actually executed):**
- **3 searches / 4 LLM calls / 4 steps.**
- It searched for the policy, then **reformulated** to find the specific "90-day" rule, then
  searched again for the newer version.
- Final answer (verbatim): *"In version 1.0 … passwords were required to be rotated every 90 days
  [S6]. In version 2.0 … the mandatory 90-day password rotation requirement was removed and
  replaced by continuous MFA verification [S4]. Employees should disregard the 90-day rotation
  requirement from the 2023 policy as version 2.0 is authoritative [S3]."*

This is exactly the behaviour the single-shot agent cannot produce: it **noticed a version
conflict, gathered evidence from multiple documents across searches, preferred the newer version,
and cited its sources.** A single top-k pass would likely surface one version and answer from it.

### 4.3 Expected benefits & the cost trade-off (🔷 Design + measured cost)

- **Where ReAct should win:** multi-hop questions (answer needs 2+ documents), questions where the
  first search is weak, and version-conflict / "compare A and B" questions. We added **6 such
  questions (q23–q28)** to the test set specifically to expose this — e.g. password policy v1↔v2,
  vendor onboarding↔offboarding, benefits + leave.
- **Where single-shot should stay competitive:** simple, single-document factual lookups (the bulk
  of q1–q22), where a second search adds latency and cost for no quality gain.
- **The cost, measured on the worked example:** ReAct used **4 LLM calls and 3 searches** for one
  answer, versus a fixed **1 and 1** for single-shot. Expect roughly **3–5× the LLM cost and
  latency per question.** The harness records `num_llm_calls` and `num_searches` per question so
  this trade-off is quantified, not guessed.

### 4.4 What is *not* yet measured

We do **not** yet have aggregate ReAct metrics (judge/ROUGE/recall over all 28 questions). The
run was blocked by external API quota limits, not by the code. **The evaluation harness, the ReAct
agent, the multi-hop test questions, and the side-by-side dashboard are all built and verified** —
only the batch execution is pending credit/quota (see below).

### 4.5 Case study — q22, the retrieval-miss question (✅ mechanism measured locally)

q22 is the single most useful example we have, because the single-shot agent **measurably failed**
it and we can show — with real retrieval, no LLM/API needed — exactly how ReAct recovers it.

> **q22:** *"If I quit my job, do I get paid for the vacation days I didn't use?"*
> **Gold document:** `HR_Leave_Policy.pdf` · **Gold answer:** unused annual leave is encashed on
> resignation (up to a 10-day cap) at last drawn basic salary.

**Root cause of the failure — a vocabulary mismatch:** the user asks in colloquial terms
("quit / vacation days / paid for"); the policy is written in formal terms
("resignation / annual leave / encashment"). The single-shot agent is locked to the user's raw
phrasing for its one and only search, so it retrieves the wrong document and cannot recover.

#### Step-by-step: single-shot vs. ReAct

| Step | Single-shot (existing) — ✅ measured | ReAct agent — ✅ retrieval measured, ⚠️ generation not run |
|---|---|---|
| 1. First search (user's raw wording) | 4 chunks, **all `Global_Employee_Handbook.pdf`** (wrong doc) | Same weak first result |
| 2. React to a weak result | ❌ No recourse — must answer from what it has | ✅ Reformulate toward policy vocabulary, search again |
| 3. Second search | — (impossible by design) | ✅ Finds `HR_Leave_Policy.pdf` |
| 4. Evidence handed to the model | Wrong doc → *"excerpts don't cover this"* | **Exact "Unused Leave Encashment" clause** |
| 5. Outcome | Retrieval **MISS**, LLM judge **1 / 5** | Retrieval **HIT** (correct answer now available) |

#### Retrieval evidence — original query vs. ReAct-style reformulations

Each row below is a real `retrieve_and_rerank` call against the live vector store (local embeddings +
BM25 + cross-encoder; no API):

| Query | Gold `HR_Leave_Policy.pdf` retrieved? | Rank |
|---|---|---|
| `"If I quit my job, do I get paid for the vacation days I didn't use?"` (single-shot's query) | ❌ **MISS** | — |
| `"unused annual leave encashment on resignation"` | ✅ Hit | **1** |
| `"vacation payout when leaving the company accrued leave"` | ✅ Hit | 2 |
| `"leave encashment last drawn salary carry-forward"` | ✅ Hit | **1** |

The recovered chunk (the one single-shot never saw) contains the answer almost verbatim:

> *"Upon resignation or termination, employees are entitled to encashment of unused annual leave
> (up to the carry-forward cap of 10 days) at their last drawn basic salary rate. Sick leave and
> casual leave are not eligible for encashment."*

#### What this proves — and its one limit

| Claim | Status |
|---|---|
| Single-shot missed the doc and scored 1/5 on q22 | ✅ Measured (prior run) |
| The miss is a reformulation problem, fully recoverable | ✅ Measured (all 3 reformulations hit; 2 at rank 1) |
| The recovered document contains the correct answer | ✅ Measured (chunk text above) |
| ReAct's final *generated* answer + judge score on q22 | ⚠️ Not run (no API quota) |

The unmeasured link is only the final generation. But the failure was in **retrieval, not
generation** — and note the paid model already scored **4/5 on q22 even with the wrong document**.
Given the *right* document, a correct, cited answer is the expected result. This is the clearest
concrete illustration of when the extra LLM-call cost of ReAct pays for itself.

---

## 5. Status — what is left

| Component | Status |
|---|---|
| ReAct agent (`rag/react_agent.py`), text-protocol tool loop | ✅ built, verified on a live multi-hop question |
| Evaluation harness with `--strategy single|react`, recall + cost telemetry | ✅ built |
| 6 multi-hop test questions (q23–q28) | ✅ added |
| Dashboard "Single vs ReAct" comparison tab | ✅ built |
| Free vs Paid measured comparison | ✅ complete (Part A) |
| **ReAct aggregate run over 28 questions** | ⏳ **pending API quota** — reruns in one command once available |
| Known pre-existing bug: `rag/generator.py:105` `NameError` in the org-clarification path | ⚠️ open, untouched |

### Why the aggregate ReAct run is pending
The free OpenRouter tier hit its **account-wide daily request cap** (~50/day without credits; our
run needs ~200), and the model we were using was decommissioned mid-effort. A paid API key was
wired in but its account had **$0 API balance** (a ChatGPT/Claude *subscription* does not fund the
*API* — separate billing). No credit was added per instruction. **To complete Part B**, either add
a small API balance (run cost < $1) or raise the OpenRouter daily cap, then run:

```bash
python -m eval.evaluate --strategy single    # baseline over all 28 questions
python -m eval.evaluate --strategy react      # ReAct over all 28 questions
streamlit run eval/dashboard.py               # open the "⚡ Single vs ReAct" tab
```

The dashboard will then render the full quality-vs-cost table, a multi-hop-only breakdown, and
per-question judge/latency/cost charts.

---

## 6. Recommendations

1. **Model tier:** Use **paid for interactive/user-facing** answering (2× faster, steadier, slightly
   better quality); **free is a solid zero-cost baseline** for batch or cost-sensitive paths.
   Retrieval quality is unaffected by this choice.
2. **Agent strategy (once Part B is measured):** Expect a **hybrid** to win — route simple factual
   questions to the cheap single-shot path and reserve the ReAct agent for multi-hop / conflict /
   "compare" questions. The `num_llm_calls` telemetry lets us price this routing precisely.
3. **Retrieval is already the strong link** (95.5% hit); the one measured miss (q22) argues for the
   ReAct "search-again" capability rather than for more retrieval tuning.
4. **Close the loop:** finish the ReAct batch run (needs a few dollars of API credit or a raised
   free cap) and fix the `generator.py:105` `NameError` before this goes near production.

---

## Appendix A — How the two approaches work (detailed)

*For teammates who aren't deep in RAG. This explains the mechanics, the step-by-step working
difference, and precisely where and why ReAct outperforms the one-shot agent.*

### A.1 One-shot RAG — the pipeline

One-shot RAG (our existing `rag/generator.py`) does **exactly one retrieval and one generation**:

```
question
  → embed the question
  → retrieve top-k chunks (dense + BM25, then rerank)
  → build one prompt: [retrieved context] + [question]
  → ONE LLM call
  → answer
```

- **One search.** The search query *is* the user's question, used verbatim.
- **One LLM call.** The model sees only what that single search returned.
- **No feedback loop.** If the search returned the wrong documents, the model cannot ask for
  different ones — it must answer from what it has (or admit it can't).

Fixed, predictable cost: **1 search + 1 LLM call** per question.

### A.2 ReAct agent — the loop

ReAct ("**Rea**son + **Act**", our `rag/react_agent.py`) turns answering into a loop where the model
decides, at each step, whether to **search again** or **finish**:

```
question
  → LOOP (up to MAX_STEPS = 5):
       Thought:  the model reasons about what it still needs
       Action:   search_knowledge_base  OR  finish
       Action Input: { "query": "<a query the MODEL writes>", ... }
       Observation: (we run the search and feed results back)
  → finish → final answer with [S#] citations
```

Two capabilities the one-shot agent structurally lacks:
1. **It writes its own search queries** — and can rewrite them if the first is weak.
2. **It can search multiple times** — gathering evidence across documents before answering.

Variable cost: **N searches + N LLM calls** (measured: ~3 searches / ~4 LLM calls on a multi-hop
question). Bounded by `MAX_STEPS` so it can't loop forever.

A real step emitted by the agent in this project:

```
Thought: Need to confirm v1.0 rotation requirement. Search for the "90-day" rule.
Action: search_knowledge_base
Action Input: {"query": "\"90-day\" password rotation", "department": "IT"}
```

### A.3 Step-by-step working difference (same question, both agents)

Using q22 — *"If I quit my job, do I get paid for the vacation days I didn't use?"*

| Stage | One-shot RAG | ReAct agent |
|---|---|---|
| **Interpret question** | none — question goes straight to search | *Thought:* "This is about leave payout on exit; I should search the leave policy." |
| **Search #1** | user's literal words → **wrong doc** (`Global_Employee_Handbook`) | same literal search → same weak result |
| **Evaluate the result** | ❌ cannot — no step for this | *Thought:* "These chunks don't mention payout/encashment; reformulate." |
| **Search #2** | ✗ impossible | `"unused annual leave encashment on resignation"` → **`HR_Leave_Policy.pdf`, rank 1** |
| **Decide to answer** | forced now | *Thought:* "I now have the encashment clause; finish." |
| **Answer** | "the excerpts don't cover this" → **judge 1/5** | grounded, cited answer from the correct clause (retrieval verified; generation pending quota) |
| **Cost** | 1 search, 1 LLM call | ~2–3 searches, ~3–4 LLM calls |

The single decisive difference is the **"Evaluate the result → reformulate"** step, which one-shot
simply does not have.

### A.4 How and why ReAct is better (and where it isn't)

**Where ReAct wins — and the reason:**

| Scenario | Why one-shot struggles | How ReAct fixes it |
|---|---|---|
| **Vocabulary mismatch** (user's words ≠ document's words) | Its only query is the user's phrasing | Reformulates toward document terminology, then re-searches (**q22, measured**) |
| **Multi-hop** (answer needs 2+ documents) | One top-k window may not hold all needed docs | Searches for each sub-topic separately and combines |
| **Version / conflict** ("what changed between v1 and v2?") | May retrieve only one version | Searches for each version, compares, prefers newest (**verified on the password-policy example**) |
| **Weak first retrieval** | Answers from wrong context, often confidently | Detects the weak result and tries again before answering |
| **Under-specified questions** | Guesses from a single pass | Can narrow scope (department/org filters) across steps |

**The mechanism in one sentence:** one-shot RAG bets everything on a single search of the user's
raw wording; ReAct treats retrieval as an *interactive* process it can steer, so a bad first search
becomes a recoverable step instead of a failed answer.

**Where ReAct is *not* better (important for balance):**
- **Simple factual lookups** (most of q1–q22): the first search already succeeds, so ReAct's extra
  steps add latency and ~3–5× LLM cost for **no quality gain**.
- **Cost & speed:** every extra step is another LLM call and more latency — meaningful at scale.
- **Model-dependence:** ReAct only helps if the model reasons well about *when* to search again; a
  weaker model can loop pointlessly or stop too early. (This is why the harness records
  `num_llm_calls`/`num_searches` — so we can confirm the extra cost actually bought better answers.)

**Practical conclusion:** ReAct is not a replacement for one-shot RAG — it is a **more capable,
more expensive tool for a specific class of hard questions.** The recommended deployment is a
**hybrid router**: cheap one-shot for simple questions, ReAct for multi-hop / conflict / retrieval-miss
questions. The per-question cost telemetry makes that routing measurable rather than guesswork.

---

*Generated from `eval/eval_results_free.json` and `eval/eval_results_paid.json` (measured), a live
ReAct execution (multi-hop worked example), and live local `retrieve_and_rerank` calls for the q22
case study (§4.5). ReAct aggregate metrics intentionally omitted until measured.*
