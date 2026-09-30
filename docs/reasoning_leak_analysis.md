# The "Reasoning Leak" Bug — Analysis, Root Cause & Fix

*A case study from the Enterprise Knowledge Assistant, written up for the team.*

---

## TL;DR

When we asked a **complex, multi-document question** (vendor onboarding **and**
offboarding), the assistant returned an answer that was actually the model's
**private chain-of-thought** — "We need to answer… Let's parse each. S3: …" —
and it was **cut off mid-sentence**. Yet the confidence badge still said
**High (1.00)**.

The root cause was **not** a bad prompt or a bad model. It was a subtle
interaction between:

1. how *reasoning models* stream their output (hidden reasoning **then** the
   final answer, separated by a boundary marker), and
2. our **token limit** (`max_tokens`) cutting the response off **before** that
   boundary was reached.

The fix is one line of intent: **turn off the model's reasoning trace for
grounded RAG answering**, plus a larger token budget and defensive parsing.

---

## 1. The symptom (what we saw)

Question (`q28`, a multi-hop question spanning two documents):

> "What steps are required when we onboard a new vendor, and what must
> Operations do when that vendor is later offboarded?"

**Single-shot answer that came back (verbatim, abridged):**

```
We need to answer: steps required when we onboard a new vendor, and what must
Operations do when that vendor is later offboarded.

We have context excerpts: S1 (Vendor Management SOP…). S2 (Operations Ecosystem
Handbook…). S3 (Vendor Onboarding SOP…). Need to extract onboarding steps from
S3 and possibly S1/S4. Offboarding steps from S1 and S2 and S4.

Let's parse each.

S3: Vendor Onboarding SOP (Procurement). It says: "…Vendor Registration…"
…
But S2 is from        ⟵ ends abruptly here (truncated)
```

Three things are wrong at once:

| # | Problem | Why it's bad |
|---|---------|--------------|
| 1 | The "answer" is the model **thinking out loud** ("We need to answer… Let's parse each"), not a user-facing answer | Unusable in a product; looks broken to an end user |
| 2 | It's **truncated mid-sentence** ("But S2 is from") | No conclusion is ever delivered |
| 3 | Confidence is still **High (1.00)** | The quality signal is wrong — a garbled answer looks trustworthy |

Crucially, **simple** questions ("How many annual leave days?") worked
perfectly. Only long, multi-source answers broke. That inconsistency was the
key clue.

---

## 2. Investigation — inspecting the raw model response

Instead of guessing, we dumped the **raw response object** from the model for a
*short* question. This is what a reasoning model actually returns:

```
finish_reason: stop
message.reasoning     : "We need to answer using only provided context, citing
                         [S1]. The context says annual leave is 21 days. So
                         answer: 21 days. Cite [S1]."      ⟵ the hidden thinking
message.content       : "21 days [S1]."                    ⟵ the clean answer
message.reasoning_details / model_extra keys: ['reasoning', 'reasoning_details']
```

**This is the "aha" moment.** The model returns **two separate fields**:

- `reasoning` — its private chain-of-thought (meant to be hidden), and
- `content` — the clean, user-facing answer.

Our code reads `content`. For short answers, `content` is clean — exactly what
we want. So where did the leak come from?

---

## 3. Root cause — truncation *before* the reasoning/answer boundary

A reasoning model generates a **single continuous token stream**:

```
[  reasoning tokens … …  ] <boundary> [  final answer tokens … ]
                                    ▲
        the provider (OpenRouter) splits the stream HERE:
        everything before the boundary  → message.reasoning
        everything after the boundary   → message.content
```

The provider can only put text in the **clean `content` field once it has seen
the boundary** that ends the reasoning phase.

Now apply our token limit (`max_tokens = 800`):

- **Short answer** → reasoning is short → the stream reaches the boundary and
  emits the answer, all within 800 tokens → provider splits correctly →
  `content` is clean. ✅
- **Complex answer (q28)** → the reasoning alone about *four* source excerpts
  ran **past 800 tokens** → generation was **cut off mid-reasoning**, **before**
  the boundary was ever produced. With no boundary to split on, the provider
  had no choice but to return the **partial reasoning in `content`** (its
  default field), truncated. ❌

So the "reasoning leak" is really a **truncation artifact**: *when a reasoning
model is cut off before it finishes thinking, its unfinished thoughts spill into
the answer field.*

And because our confidence score was based on retrieval similarity (which was
genuinely high — the right documents *were* found), the broken answer still
scored **High**.

---

## 4. Why it hit the ReAct agent too

The ReAct agent uses the *same* underlying model call, so it inherited the same
truncation behavior — its intermediate `Thought:` steps competed for the same
token budget, and on q28 its searches also drifted (it retrieved *Client
Onboarding* and *Employee Onboarding* SOPs instead of the *Vendor* Onboarding
SOP). Fixing the model-call layer benefits **both** strategies.

### A related leak in the ReAct loop

Verifying the fix surfaced a **second, related** scaffold leak in the ReAct
agent. When the agent finishes, it emits its answer as JSON:
`Action Input: {"answer": "<final answer>"}`. The free model sometimes produces
**invalid JSON** — an unquoted value like `{"answer": The documents…}`. Our
parser couldn't read it, and the fallback dumped the **raw ReAct scaffold**
into the user's answer:

```
Thought: I have searched for vendor onboarding…
Action: finish
Action Input: {"answer": The documents indicate that vendor onboarding…}
```

Same class of bug — internal scaffolding leaking into a user-facing answer. The
fix salvages the answer: it first extracts the text after the `"answer":`
marker, and failing that strips the `Thought:/Action:/Action Input:` lines, so
the user only ever sees the answer text.

---

## 5. The fix

Three complementary changes, all at the single model-call chokepoint
(`rag/llm_client.py`), so every caller benefits:

### (a) Turn off the reasoning trace for grounded RAG *(primary fix)*

Grounded question-answering over retrieved text doesn't need an extended
chain-of-thought — the reasoning is what *caused* the overflow. We disable it
via the provider's unified parameter:

```python
extra_body={"reasoning": {"enabled": False}}
```

This eliminates the reasoning phase entirely, so:
- there is no reasoning to overflow into `content`,
- the **whole** token budget goes to the actual answer, and
- responses are faster and more concise.

### (b) Raise the token budget (belt-and-suspenders)

Even with reasoning off, a genuinely long multi-document answer needs room.
We raised the limit so complex answers finish cleanly (`finish_reason: stop`,
not `length`).

### (c) Defensive parsing

If a future model still emits reasoning inline (e.g. wrapped in `<think>…</think>`
tags), strip it; and if `content` ever comes back empty while a `reasoning`
field is populated, fall back to it rather than returning nothing.

---

## 6. Verification

Same two-document question, **reasoning disabled**, small budget — the exact
scenario that broke before:

```
finish_reason: stop
reasoning field length: 0
CONTENT: "Vendor onboarding steps: vendor completes registration form; Finance
runs sanctions screening. [S1]
Operations must revoke access within 5 business days and confirm data deletion
within 30 days at offboarding. [S2]"
```

Clean, complete, correctly cited, not truncated — and it fit in a **fraction**
of the tokens the leaked reasoning had consumed.

---

## 7. Lessons for the presentation

- **Reasoning models return two channels** (`reasoning` + `content`). If you
  read the wrong one — or cut the stream off before the split — you get the
  model's scratchpad instead of its answer.
- **`max_tokens` is a correctness knob, not just a cost knob.** Truncating a
  reasoning model mid-thought produces *qualitatively* broken output, not just
  a shorter answer.
- **Confidence must reflect the answer, not just retrieval.** A high-similarity
  retrieval can still yield a garbled answer; our calibration now downgrades
  refusals/hedges, and this class of failure is why that matters.
- **Inspect the raw API response before theorizing.** One dump of the response
  object turned an ambiguous "the model is dumping its thoughts" into a precise,
  one-line fix.
