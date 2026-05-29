"""
Streamlit UI for the RAG chatbot.

Provides a chat interface with expandable source citations, conversation
memory, and a sidebar showing project statistics and sample Q/A pairs.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.chatbot import RAGChatbot
from src.vector_store import get_chroma_client, get_embedding_function

RAW_DATA_DIR = Path(__file__).parent / "data" / "raw"
QA_DATASET_PATH = Path(__file__).parent / "data" / "qa_dataset.csv"
BLOG_URL = "https://addyosmani.com/blog/"


@st.cache_data(show_spinner=False)
def _load_stats() -> dict:
    """Gather project statistics from the data directory and vector store."""
    posts = len(list(RAW_DATA_DIR.glob("*.json")))

    qa_count = 0
    if QA_DATASET_PATH.exists():
        qa_count = len(pd.read_csv(QA_DATASET_PATH))

    chunk_count = 0
    try:
        client = get_chroma_client()
        ef = get_embedding_function()
        chunk_count = client.get_collection("blog_chunks", embedding_function=ef).count()
    except Exception:
        pass

    return {"posts": posts, "qa_pairs": qa_count, "chunks": chunk_count}


@st.cache_data(show_spinner=False)
def _load_qa_dataset() -> pd.DataFrame:
    """Load the full Q/A dataset for the sample viewer."""
    if not QA_DATASET_PATH.exists():
        return pd.DataFrame(columns=["question", "answer", "source_page"])
    return pd.read_csv(QA_DATASET_PATH)


def _get_chatbot() -> RAGChatbot:
    """Return the session-scoped RAGChatbot, creating it on first call."""
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatbot()
    return st.session_state.chatbot


def _render_sidebar():
    """Render sidebar with project info, statistics, and sample Q/A viewer."""
    with st.sidebar:
        st.header("About")
        st.markdown(
            f"RAG chatbot powered by **hybrid retrieval** over "
            f"[Addy Osmani's blog]({BLOG_URL}).\n\n"
            "Ask about software engineering, AI, "
            "coding agents, and leadership."
        )

        st.divider()

        st.subheader("Statistics")
        stats = _load_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Posts", stats["posts"])
        col2.metric("Q/A", stats["qa_pairs"])
        col3.metric("Chunks", stats["chunks"])

        st.divider()

        st.subheader("Sample Q/A Pairs")
        if st.button("Shuffle"):
            st.session_state.pop("sample_qa", None)

        qa_df = _load_qa_dataset()
        if not qa_df.empty:
            if "sample_qa" not in st.session_state:
                st.session_state.sample_qa = qa_df.sample(
                    n=min(5, len(qa_df))
                ).to_dict("records")

            for pair in st.session_state.sample_qa:
                q = pair["question"]
                label = (q[:80] + "...") if len(q) > 80 else q
                with st.expander(label):
                    st.markdown(f"**A:** {pair['answer']}")

        st.divider()

        if st.button("Clear Chat", type="primary", use_container_width=True):
            _get_chatbot().reset()
            st.session_state.messages = []
            st.rerun()


def _render_sources(sources: list[dict], strategy: str):
    """Show source citations as an expandable section below an answer."""
    if not sources:
        return

    label = "Q/A match" if strategy == "qa" else f"{len(sources)} source(s)"
    with st.expander(f"Sources — {label}"):
        for source in sources:
            st.markdown(f"- [{source['title']}]({source['url']})")


def _render_chat():
    """Display chat history and process new user input."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Hi! Ask me anything about Addy Osmani's blog posts "
                "on software engineering, AI, and coding agents."
            )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"], msg.get("strategy", ""))

    prompt = st.chat_input("Ask about Addy Osmani's blog posts...")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = _get_chatbot().chat(prompt)
        except Exception as exc:
            st.error(f"Failed to generate a response: {exc}")
            return

        st.markdown(result["answer"])
        _render_sources(result["sources"], result["strategy"])

    st.session_state.messages.extend([
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "strategy": result["strategy"],
        },
    ])


def main():
    st.set_page_config(
        page_title="Addy Osmani Blog RAG Chatbot",
        page_icon="💬",
        layout="centered",
    )

    st.title("💬 RAG Chatbot for Addy's Blog")
    st.caption("Powered by hybrid Q/A + vector retrieval and Ollama")

    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
