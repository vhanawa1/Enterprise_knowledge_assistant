"""
ReAct agent -- an alternative answer-generation strategy.

Where rag.generator.generate_answer does ONE retrieval followed by ONE LLM
call, this agent reasons in a Thought -> Action -> Observation loop and can
search the knowledge base multiple times: reformulating the query when the
first search misses, or gathering evidence from several documents for a
multi-hop question, before it commits to an answer.

It is a drop-in alternative for evaluation: `answer_with_react` returns the
SAME result dict shape as generate_answer
    {answer, citations, confidence, confidence_score, chunks_used}
plus agent-specific telemetry (num_llm_calls, num_searches, steps, trace) so
eval/evaluate.py can score both strategies with identical metrics.

Tool protocol: GLM (the free model) has no native tool-calling endpoint on
OpenRouter, so tools are driven via a text protocol the model emits and we
parse -- verified to work reliably with a `stop=["Observation:"]` guard.
"""
from __future__ import annotations

import json
import re
from typing import Any

from rag.llm_client import chat_complete
from rag.retriever import retrieve_and_rerank
from rag.generator import _confidence_label, assess_confidence
from rag.vector_store import VectorStore

MAX_STEPS = 5           # hard cap on Thought/Action iterations (bounds cost)
OBS_SNIPPET_CHARS = 700  # truncate each result snippet in observations. Kept
                         # generous enough that the model can read a chunk's
                         # actual content in one observation, rather than
                         # burning extra searches trying to "see more".

SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant that answers employee \
questions using ONLY internal company documents you retrieve via tools. You work in a \
loop of Thought -> Action -> Observation.

AVAILABLE ACTIONS:
- search_knowledge_base: search the documents. Action Input JSON:
    {"query": "<search text>", "organization": "<org name or omit>"}
- finish: give the final answer. Action Input JSON:
    {"answer": "<final answer, grounded ONLY in observed sources, with [S#] citations>"}

RULES:
1. Ground every claim in retrieved context. Never use outside knowledge.
2. Cite sources with the bracket tags shown in observations, e.g. [S1], [S2].
3. If your first search is weak, REFORMULATE and search again. For questions that \
span multiple topics or documents, search for each part before finishing. A topic \
may live in an unexpected place (e.g. vendor onboarding is owned by Procurement, not \
Operations), so search by TOPIC keywords -- do not assume which team owns a document.
4. If, after searching, the documents do not contain enough information, finish and \
say so explicitly rather than guessing -- suggest who to contact.
5. If retrieved excerpts conflict (e.g. different policy versions), note the conflict \
and prefer the most recent version.

RESPONSE FORMAT -- respond with EXACTLY one step, nothing else:
Thought: <your reasoning>
Action: <search_knowledge_base OR finish>
Action Input: <a single-line JSON object>

Do NOT write an Observation yourself; stop after Action Input and wait for it."""

def _build_where(organization: str | None) -> dict | None:
    """Only ever filter by organization (used to disambiguate between separate
    company entities). We deliberately do NOT filter by department: a guessed
    department is a hard metadata filter that silently excludes the correct
    document when the guess is wrong (e.g. vendor onboarding is owned by
    Procurement, not Operations) -- the reranker handles relevance instead.
    See docs/reasoning_leak_analysis.md's sibling note on retrieval drift."""
    if organization:
        return {"organization": organization}
    return None


def _parse_step(text: str) -> tuple[str, dict]:
    """Extract (action, action_input_dict) from a model step. Returns
    ("", {}) if it can't be parsed."""
    action_match = re.search(r"Action:\s*(\w+)", text)
    if not action_match:
        return "", {}
    action = action_match.group(1).strip()

    # Grab everything after "Action Input:" and pull the first JSON object.
    input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
    args: dict = {}
    if input_match:
        raw = input_match.group(1).strip()
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace:
            try:
                args = json.loads(brace.group(0))
            except json.JSONDecodeError:
                args = {}
    return action, args


def _finish_answer(args: dict, step_text: str) -> str:
    """Extract the final answer from a `finish` step.

    Ideally the answer is the parsed JSON `answer` field. But the free model
    sometimes emits invalid JSON (e.g. an unquoted value: {"answer": The docs...}),
    so `args` comes back empty. Rather than dumping the raw ReAct scaffold
    ("Thought: ... Action: finish Action Input: ...") into the user's answer,
    salvage the answer text: first from an `"answer":` marker, else by stripping
    the Thought/Action/Action Input/Observation scaffold lines."""
    ans = args.get("answer")
    if ans:
        return str(ans).strip()
    m = re.search(r'"answer"\s*:\s*(.+)', step_text, re.DOTALL)
    if m:
        salvaged = m.group(1).strip().lstrip('"').rstrip('}').strip().rstrip('"').strip()
        if salvaged:
            return salvaged
    lines = [ln for ln in step_text.splitlines()
             if not re.match(r'\s*(Thought|Action|Action Input|Observation)\s*:', ln)]
    return "\n".join(lines).strip() or step_text.strip()


def _register(chunks: list[dict], registry: dict, order: list) -> None:
    """Assign each newly-seen chunk a stable [S#] tag for the whole episode,
    so the model's citations line up with the final citation list."""
    for c in chunks:
        cid = c["id"]
        if cid not in registry:
            order.append(cid)
            registry[cid] = {"chunk": c, "tag": f"S{len(order)}"}


def _format_observation(chunks: list[dict], registry: dict) -> str:
    if not chunks:
        return "No results found for that query. Try different search terms."
    lines = []
    for c in chunks:
        entry = registry[c["id"]]
        m = c["metadata"]
        header = (f"[{entry['tag']}] {m.get('title', m.get('source'))} | "
                  f"Org: {m.get('organization')} | Dept: {m.get('department')} | "
                  f"Version: {m.get('version')} | Page: {m.get('page')}")
        snippet = c["text"][:OBS_SNIPPET_CHARS].replace("\n", " ")
        lines.append(f"{header}\n{snippet}")
    return "\n\n".join(lines)


def _build_result(answer: str, registry: dict, order: list, *,
                  num_llm_calls: int, num_searches: int, steps: int,
                  trace: list[str]) -> dict[str, Any]:
    chunks_used = [registry[cid]["chunk"] for cid in order]
    top_score = max((c.get("final_score", c.get("score", 0)) for c in chunks_used), default=0.0)
    citations = [
        {
            "tag": registry[cid]["tag"],
            "source": registry[cid]["chunk"]["metadata"].get(
                "title", registry[cid]["chunk"]["metadata"].get("source")),
            "organization": registry[cid]["chunk"]["metadata"].get("organization"),
            "department": registry[cid]["chunk"]["metadata"].get("department"),
            "page": registry[cid]["chunk"]["metadata"].get("page"),
            "version": registry[cid]["chunk"]["metadata"].get("version"),
            "score": round(registry[cid]["chunk"].get("final_score",
                           registry[cid]["chunk"].get("score", 0)), 3),
        }
        for cid in order
    ]
    if chunks_used:
        confidence, confidence_score = assess_confidence(answer, top_score)
    else:
        confidence, confidence_score = "Low", 0.0
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "chunks_used": chunks_used,
        # --- agent telemetry (ignored by generate_answer callers) ---
        "num_llm_calls": num_llm_calls,
        "num_searches": num_searches,
        "steps": steps,
        "trace": trace,
    }


def answer_with_react(
    question: str,
    store: VectorStore,
    chat_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run the ReAct loop and return a generate_answer-shaped result dict."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-6:])
    messages.append({"role": "user", "content": f"Question: {question}\nBegin."})

    registry: dict[str, dict] = {}
    order: list[str] = []
    trace: list[str] = []
    num_llm_calls = 0
    num_searches = 0

    for step in range(1, MAX_STEPS + 1):
        step_text = chat_complete(messages, temperature=0.0, max_tokens=500,
                                  stop=["Observation:"]) or ""
        num_llm_calls += 1
        trace.append(step_text)
        action, args = _parse_step(step_text)

        if action == "finish":
            answer = _finish_answer(args, step_text)
            return _build_result(answer, registry, order,
                                 num_llm_calls=num_llm_calls, num_searches=num_searches,
                                 steps=step, trace=trace)

        if action == "search_knowledge_base":
            num_searches += 1
            query = args.get("query", question)
            where = _build_where(args.get("organization"))
            chunks = retrieve_and_rerank(query, store, where=where)
            _register(chunks, registry, order)
            observation = _format_observation(chunks, registry)
            messages.append({"role": "assistant", "content": step_text})
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            continue

        # Unparseable / unknown action -> nudge the model back on protocol.
        messages.append({"role": "assistant", "content": step_text})
        messages.append({"role": "user", "content": (
            "Observation: Could not parse your action. Respond with a valid "
            "'Action: search_knowledge_base' or 'Action: finish' and a single-line "
            "JSON 'Action Input:'.")})

    # Step budget exhausted -> force a grounded answer from what we have.
    messages.append({"role": "user", "content": (
        "You have reached the search limit. Using ONLY the observations above, "
        "give your best grounded answer now with [S#] citations, or say the "
        "documents don't contain enough information.")})
    final = chat_complete(messages, temperature=0.0, max_tokens=1200) or ""
    num_llm_calls += 1
    trace.append(final)
    return _build_result(final.strip(), registry, order,
                         num_llm_calls=num_llm_calls, num_searches=num_searches,
                         steps=MAX_STEPS, trace=trace)
