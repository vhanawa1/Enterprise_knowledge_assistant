"""
Answer generation: builds a grounded prompt from reranked chunks, calls the
LLM, and returns a structured response with citations and a confidence
label. Also supports document summarization and contextual follow-ups by
accepting prior chat history.
"""
from __future__ import annotations

from typing import Any

import config
from rag.llm_client import chat_complete

SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. Answer the employee's \
question using ONLY the information in the provided context excerpts, which come from \
internal company documents. Follow these rules strictly:

1. Ground every claim in the provided context. Do not use outside knowledge.
2. After each factual statement, cite the source using the bracket tag given with \
   that excerpt, e.g. [S1], [S2]. Use multiple tags if a statement draws on multiple sources.
3. If the context does not contain enough information to answer confidently, say so \
   explicitly rather than guessing, and suggest what the employee should check or who \
   to contact.
4. If different excerpts conflict (e.g. different policy versions), point out the \
   conflict and prefer the most recent version if version/date metadata is available.
5. Be concise and directly answer the question first, then add necessary detail.
6. If the user asks a follow-up question, use the conversation history to resolve \
   pronouns and implicit references (e.g. "what about for contractors?").
"""


def _format_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        m = c["metadata"]
        header = (f"[S{i}] Source: {m.get('title', m.get('source'))} | "
            f"Org: {m.get('organization')} | Dept: {m.get('department')} | "
            f"Version: {m.get('version')} | Page: {m.get('page')}")
        blocks.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def _confidence_label(top_score: float) -> str:
    if top_score >= config.CONFIDENCE_HIGH:
        return "High"
    elif top_score >= config.CONFIDENCE_MEDIUM:
        return "Medium"
    return "Low"


# --- Groundedness-aware confidence -------------------------------------------
# Retrieval similarity alone is a poor confidence signal: the reranker can score
# an irrelevant-but-topically-similar chunk highly, so a model that then refuses
# ("the excerpts don't cover this") would still report High. We downgrade the
# confidence based on what the ANSWER actually says, so a failed/hedged answer
# no longer masquerades as High -- and the gap between a single-shot miss (Low)
# and a deeper ReAct answer (Medium/High) becomes a visible reason to escalate.

# The model couldn't answer at all from the retrieved context.
_REFUSAL_MARKERS = (
    "couldn't find", "could not find", "do not contain", "does not contain",
    "don't contain", "not contain enough", "don't have enough", "do not have enough",
    "no relevant information", "not enough information", "cannot answer",
)
# The model answered only partially / with explicit uncertainty.
_HEDGE_MARKERS = (
    "cannot confirm", "can't confirm", "not fully visible", "not fully shown",
    "consult the complete", "consult the full", "for a definitive answer",
    "not clear from", "unable to determine", "partially", "may not be complete",
)


def _is_refusal(answer: str) -> bool:
    t = (answer or "").lower()
    return any(m in t for m in _REFUSAL_MARKERS)


def _is_hedge(answer: str) -> bool:
    t = (answer or "").lower()
    return any(m in t for m in _HEDGE_MARKERS)


def assess_confidence(answer: str, top_score: float) -> tuple[str, float]:
    """Return (label, score) reflecting answer groundedness, not just retrieval.
    Refusal caps confidence into the Low band; a hedged/partial answer caps it
    into the Medium band; otherwise the retrieval score stands."""
    score = top_score
    if _is_refusal(answer):
        score = min(score, config.CONFIDENCE_MEDIUM - 0.05)  # force Low
    elif _is_hedge(answer):
        score = min(score, config.CONFIDENCE_HIGH - 0.05)    # cap at Medium
    return _confidence_label(score), round(score, 3)

def _distinct_organizations(chunks: list[dict]) -> list[str]:
    return sorted(set(c["metadata"].get("organization", "Unknown") for c in chunks))

def _question_names_an_org(question: str, orgs: list[str]) -> bool:
    """True if the question already names one of the candidate orgs, so
    we don't ask for clarification when the person has already been
    specific about which entity they mean."""
    q = question.lower()
    for org in orgs:
        full_lower = org.lower()
        # First two words as the disambiguating signal (e.g. "northwind
        # analytics" vs "northwind labs") -- a single shared first word
        # like "northwind" wouldn't actually distinguish them.
        two_words = " ".join(org.lower().replace(",", "").split()[:2])
        if two_words in q or full_lower in q:
            return True
    return False

def _needs_org_clarification(question: str, chunks: list[dict]) -> tuple[bool, list[str]]:
    """Deterministic check -- not left to the LLM's judgment, since
    smaller/free models are less reliable at noticing this kind of
    cross-document conflict on their own."""
    orgs = _distinct_organizations(chunks)
    if len(orgs) <= 1:
        return False, orgs
    if _question_names_an_org(question, orgs):
        return False, orgs
    return True, orgs

def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Returns {"answer", "citations", "confidence", "confidence_score", "chunks_used"}"""
    if not chunks:
        return {
            "answer": ("I couldn't find any relevant information in the indexed documents "
                       "to answer this question. Try rephrasing, or check with the relevant "
                       "department directly."),
            "citations": [],
            "confidence": "Low",
            "confidence_score": 0.0,
            "chunks_used": [],
        }
    needs_clarification, orgs = _needs_org_clarification(question, chunks)
    if needs_clarification:
        org_list = "\n".join(f"- {o}" for o in orgs)
        return {
            "answer": (
                f"This question could relate to more than one organization in the knowledge base:\n\n"
                f"{org_list}\n\n"
                f"Could you clarify which one you mean? (You can also use the sidebar filters to "
                f"scope your question to one organization.)"
            ),
            "citations": [
                {
                    "tag": f"S{i+1}",
                    "source": c["metadata"].get("title", c["metadata"].get("source")),
                    "organization": c["metadata"].get("organization"),
                    "department": c["metadata"].get("department"),
                    "page": c["metadata"].get("page"),
                    "version": c["metadata"].get("version"),
                    "score": round(c.get("final_score", c.get("score", 0)), 3),
                }
                for i, c in enumerate(chunks)
            ],
            "confidence": "Low",
            "confidence_score": 0.0,
            "chunks_used": chunks,
        }
    context = _format_context(chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        # include last few turns for contextual follow-ups
        messages.extend(chat_history[-6:])

    user_prompt = f"CONTEXT EXCERPTS:\n\n{context}\n\nQUESTION: {question}"
    messages.append({"role": "user", "content": user_prompt})

    answer_text = chat_complete(messages, temperature=0.2, max_tokens=1500)

    top_score = max((c.get("final_score", c.get("score", 0)) for c in chunks), default=0)
    citations = [
        {
            "tag": f"S{i+1}",
            "source": c["metadata"].get("title", c["metadata"].get("source")),
            "department": c["metadata"].get("department"),
            "page": c["metadata"].get("page"),
            "version": c["metadata"].get("version"),
            "score": round(c.get("final_score", c.get("score", 0)), 3),
        }
        for i, c in enumerate(chunks)
    ]

    confidence, confidence_score = assess_confidence(answer_text, top_score)
    return {
        "answer": answer_text,
        "citations": citations,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "chunks_used": chunks,
    }


def summarize_document(source_name: str, full_text: str) -> str:
    messages = [
        {"role": "system", "content": ("You summarize internal company documents for busy "
                                        "employees. Produce a concise summary with: 1) a one-line "
                                        "purpose, 2) key points as bullets, 3) any dates/numbers/ "
                                        "thresholds mentioned (leave counts, deadlines, amounts). "
                                        "Do not add information not present in the text.")},
        {"role": "user", "content": f"Document: {source_name}\n\n{full_text[:12000]}"},
    ]
    return chat_complete(messages, temperature=0.2, max_tokens=500)
