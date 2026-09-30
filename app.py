"""
Enterprise Knowledge Assistant -- Streamlit chatbot UI.

Run:
    streamlit run app.py
"""
from multiprocessing import Value
import time
import json
from pathlib import Path
from datetime import datetime

import streamlit as st

import config
from rag.vector_store import VectorStore
from rag.retriever import retrieve_and_rerank
from rag.generator import generate_answer, _is_refusal
from rag.react_agent import answer_with_react, MAX_STEPS

st.set_page_config(
        #page_title="Enterprise Knowledge Assistant", 
        page_icon="🧠", 
        layout="wide",
        initial_sidebar_state="expanded"
        )

CONFIDENCE_COLORS = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}

def needs_escalation(result: dict) -> bool:
    """True when the cheap single-shot answer is weak enough to be worth
    offering the (more expensive) ReAct agent: low confidence, nothing
    retrieved, or an explicit 'couldn't find it' refusal."""
    return (
        result.get("confidence") == "Low"
        or not result.get("chunks_used")
        or _is_refusal(result.get("answer", ""))
    )


def run_pending_react(store):
    """If the user accepted the ReAct offer, run the agent (multi-step) and
    append its answer as a new assistant turn."""
    question = st.session_state.pop("run_react_for", None)
    if not question:
        return
    with st.spinner("🔷 ReAct agent — reasoning and searching across multiple steps…"):
        t0 = time.time()
        result = answer_with_react(question, store)
        latency = round(time.time() - t0, 2)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "citations": result.get("citations", []),
        "confidence": result.get("confidence"),
        "confidence_score": result.get("confidence_score"),
        "strategy": "react",
        "num_llm_calls": result.get("num_llm_calls"),
        "num_searches": result.get("num_searches"),
        "latency": latency,
    })
    try:
        save_history(st.session_state.messages)
    except Exception:
        pass
    st.rerun()


def render_escalation_offer(store):
    """Show the 'try ReAct?' panel under the latest answer, but only when that
    answer was a single-shot result flagged as weak and not yet acted on."""
    msgs = st.session_state.get("messages", [])
    if not msgs:
        return
    last = msgs[-1]
    if (last.get("role") != "assistant"
            or last.get("strategy") != "single"
            or not last.get("escalatable")
            or last.get("escalation_declined")):
        return
    question = last.get("question")
    if not question:
        return

    st.info(
        "🤔 **That answer looks weak** — low confidence or no matching document. "
        "I can escalate to the **ReAct agent**, which rewrites the query and searches the "
        "knowledge base several times instead of once.\n\n"
        f"**Cost of escalating:** up to ~{MAX_STEPS} reason+search steps "
        f"(≈{MAX_STEPS} LLM calls and searches) versus the single call just used — slower and "
        "higher token cost, but much stronger on multi-hop and vocabulary-mismatch questions."
    )
    c1, c2, _ = st.columns([1.2, 1, 3])
    with c1:
        if st.button("🔷 Yes, use ReAct agent", key="escalate_yes", type="primary"):
            st.session_state.run_react_for = question
            st.rerun()
    with c2:
        if st.button("No thanks", key="escalate_no"):
            last["escalation_declined"] = True
            try:
                save_history(st.session_state.messages)
            except Exception:
                pass
            st.rerun()


st.html(
    """
    <style>
    .chat-citation {
        font-size: 0.8rem !important; /* Makes it smaller than standard chat font */
        color: #888888;              /* Soft gray look */
        display: block;
        margin-top: 2px;
    }
    </style>
    """
)


st.markdown(
    """
    <style>
      /* global app text */
      [data-testid="stAppViewContainer"] { font-size:13px; }
      /* markdown blocks */
      .stMarkdown p { font-size:13px; }
      /* chat message text (Streamlit testid/class may vary by version) */
      [data-testid="stChatMessageText"] { font-size:13px; }

      [data-testid="stMarkdownContainer"] { font-size:13px; }

      [data-testid="stSidebarHeader"] {
        height: 0rem !important; /* Forces the header div to remain small */
        min-height: 0rem !important;}

         hr {
        margin-top: 2px !important;    /* Padding BEFORE the divider */
        margin-bottom: 2px !important; /* Padding AFTER the divider */
            }
      
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_store():
    return VectorStore()


# Persist chat history between sessions by saving to a JSON file under data/
HISTORY_PATH = Path(config.BASE_DIR) / "data" / "chat_history.json"
ARCHIVE_PATH = Path(config.BASE_DIR) / "data" / "chat_archives.json"


def load_history():
    """Load chat history from disk. Returns a list of message dicts.

    If loading fails, returns an empty list.
    """
    try:
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception as e:
        # Keep UI simple: surface a small warning but continue with empty history
        try:
            st.warning(f"Could not load chat history: {e}")
        except Exception:
            pass
    return []


def save_history(messages):
    """Save chat history (list of messages) to disk.

    Creates parent directories if needed. Swallows errors to avoid breaking the UI.
    """
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
            json.dump(messages, fh, ensure_ascii=False, indent=2)
    except Exception:
        # Non-fatal: do not raise — the app should continue even if persistence fails
        return


def load_archives():
    """Load archived conversations from disk. Returns a list of archive entries.

    Each entry is a dict: {"timestamp": <float>, "messages": [...]}
    """
    try:
        if ARCHIVE_PATH.exists():
            with open(ARCHIVE_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        try:
            st.warning("Could not load archived chats")
        except Exception:
            pass
    return []


def save_archives(archives):
    try:
        ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ARCHIVE_PATH, "w", encoding="utf-8") as fh:
            json.dump(archives, fh, ensure_ascii=False, indent=2)
    except Exception:
        return


def archive_current_conversation():
    """Append the current session messages to the archives with a timestamp."""
    try:
        archives = load_archives()
        entry = {"timestamp": time.time(), "messages": st.session_state.get("messages", [])}
        archives.append(entry)
        save_archives(archives)
    except Exception:
        return


def build_where_clause(departments, sensitivities):
    conditions = []
    if departments:
        conditions.append({"department": {"$in": departments}})
    if sensitivities:
        conditions.append({"sensitivity": {"$in": sensitivities}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def main():
    #st.title("🧠 Enterprise Knowledge Assistant")
    #st.caption("Ask questions in plain English. Answers are grounded in your organization's documents, with citations.")

    store = get_store()
    doc_summary = store.get_all_documents_summary()

    

    # ---------------- Sidebar: filters + index status ----------------
    with st.sidebar:
        st.header("🧠 Enterprise Knowledge Assistant", divider="rainbow")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("➕ New chat", key="new_chat"):
                try:
                    archive_current_conversation()
                except Exception:
                    pass
                st.session_state.messages = []
                try:
                    save_history([])
                except Exception:
                    pass
                st.rerun()
        with col2:
            if st.button("🗑️ Clear chat", key="clear_chat_sidebar"):
                st.session_state.messages = []
                try:
                    save_history([])
                except Exception:
                    pass
                st.rerun()
                    
        st.header("💬 Recent chats")
        archives = load_archives()
        if archives:
            recent = archives[-5:][::-1]
            for idx, a in enumerate(recent):
                ts = datetime.fromtimestamp(a.get("timestamp", 0)).strftime("%Y-%m-%d")
                preview = ""
                if a.get("messages"):
                    for m in a["messages"]:
                        if m.get("role") == "user":
                            preview = m.get("content", "")[:120]
                            break
                    if not preview and a["messages"]:
                        preview = a["messages"][0].get("content", "")[:120]
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.write(f"**{ts}** — {preview}")
                with col2:
                    if st.button("Open", key=f"open_archive_{idx}"):
                        st.session_state.messages = a.get("messages", [])
                        try:
                            save_history(st.session_state.messages)
                        except Exception:
                            pass
                        st.rerun()
        else:
            st.write("No recent chats")
        
        st.divider()
        st.subheader("⚙️ Filters")
        depts = sorted({d["department"] for d in doc_summary if d["department"]})
        sens = sorted({d["sensitivity"] for d in doc_summary if d["sensitivity"]})

        selected_depts = st.multiselect("Department", options=depts or config.DEPARTMENTS)
        selected_sens = st.multiselect("Sensitivity", options=sens or config.SENSITIVITY_LEVELS)

        st.divider()
        if config.RAG_MODE == "free":
            st.info(f"**FREE mode**\n\nChat: `{config.OPENROUTER_CHAT_MODEL}` (OpenRouter)\n\nEmbeddings: `{config.LOCAL_EMBEDDING_MODEL}` (local)")
        else:
            provider_label = "Azure OpenAI" if config.PROVIDER == "azure_openai" else "OpenAI"
            st.success(f" **PAID mode**\n\nChat: `{config.CHAT_MODEL if config.PROVIDER != 'azure_openai' else config.AZURE_CHAT_DEPLOYMENT}` ({provider_label})")
        #st.caption("Switch modes by setting RAG_MODE=free or RAG_MODE=paid in .env, then restart the app.")
        st.divider()
        
        
        st.subheader("📚 Knowledge Base")
        col1, col2 = st.columns([1, 1])
        with col1:
             st.metric("Indexed chunks", store.count(), border=True)
        with col2:
             st.metric("Indexed documents", len(doc_summary), border=True)
        with st.expander("View indexed documents"):
            for d in doc_summary:
                st.write(f"**{d['title']}** — {d['department']} / {d['sensitivity']} (v{d['version']})")

        
        


    if store.count() == 0:
        st.warning(
            "No documents are indexed yet. Run the ingestion pipeline first:\n\n"
            "`python -m ingestion.ingest`\n\n"
            "(Sample HR/IT/Operations documents are provided in `data/sample_docs/`.)"
        )
        return

    # ---------------- Chat state ----------------
    if "messages" not in st.session_state:
        # load persisted history (if any) so history survives app restarts
        st.session_state.messages = load_history()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_citations(msg.get("citations"), msg.get("confidence"), msg.get("confidence_score"))
                render_answer_meta(msg)

    # If the user accepted a ReAct escalation on the previous run, execute it now.
    run_pending_react(store)
    # Offer to escalate the latest answer to ReAct if it came back weak.
    render_escalation_offer(store)

    question = st.chat_input("Ask about HR policy, IT support, onboarding, or any indexed document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        # persist after user adds a message
        try:
            save_history(st.session_state.messages)
        except Exception:
            pass
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant documents and generating answer..."):
                where = build_where_clause(selected_depts, selected_sens)
                t0 = time.time()
                chunks = retrieve_and_rerank(question, store, where=where)

                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                    if m["role"] in ("user", "assistant")
                ]
                result = generate_answer(question, chunks, chat_history=history)
                latency = round(time.time() - t0, 2)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "confidence": result["confidence"],
            "confidence_score": result["confidence_score"],
            "strategy": "single",
            "latency": latency,
            "sources_count": len(chunks),
            # Remember the question + whether this weak answer is worth escalating,
            # so the ReAct offer can render (and re-run the agent) after the rerun.
            "question": question,
            "escalatable": needs_escalation(result),
        })
        # persist after assistant reply
        try:
            save_history(st.session_state.messages)
        except Exception:
            pass
        st.rerun()


def render_answer_meta(msg: dict):
    """Small caption line under an assistant answer: which strategy produced it
    and what it cost (single-shot = 1 call; ReAct = its measured calls/searches)."""
    parts = []
    if msg.get("strategy") == "react":
        parts.append(
            f"🔷 ReAct agent · {msg.get('num_llm_calls', '?')} LLM calls · "
            f"{msg.get('num_searches', '?')} searches"
        )
    elif msg.get("strategy") == "single":
        parts.append("⚡ Single-shot · 1 LLM call")
    if msg.get("latency") is not None:
        parts.append(f"⏱️ {msg['latency']}s")
    if msg.get("sources_count") is not None:
        parts.append(f"{msg['sources_count']} sources retrieved")
    if parts:
        st.caption(" · ".join(parts))


def render_citations(citations, confidence, confidence_score):
    if confidence:
        icon = CONFIDENCE_COLORS.get(confidence, "⚪")
        score_txt = f" ({confidence_score:.2f})" if confidence_score is not None else ""
        st.caption(f"{icon} **Confidence: {confidence}**{score_txt}")
    if citations:
        with st.expander(f"📎 Sources ({len(citations)})"):
            for c in citations:
                st.markdown(
                    f"**[{c['tag']}]** {c['source']} — {c['department']}, "
                    f"page {c['page']}, v{c['version']}  \n"
                    f"relevance score: `{c['score']}`"
                )


    


if __name__ == "__main__":
    main()
