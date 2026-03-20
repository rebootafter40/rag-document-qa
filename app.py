"""
app.py — Streamlit frontend for the RAG Document Q&A app.
"""

import streamlit as st
from src.retriever import ingest_pdf, retrieve
from src.qa_chain import ask
from src.vector_store import clear_collection
import tempfile
import os


st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Q&A")
st.markdown("Upload a PDF and ask questions about it. Answers are grounded in your document with source citations.")

# --- Sidebar ---
with st.sidebar:
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file and "processed_file" not in st.session_state:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document... This may take a minute."):
                # Save uploaded file to a temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                # Clear old data and ingest new document
                clear_collection()
                num_chunks = ingest_pdf(tmp_path)

                # Clean up temp file
                os.unlink(tmp_path)

                # Save state so we don't reprocess
                st.session_state["processed_file"] = uploaded_file.name
                st.session_state["num_chunks"] = num_chunks
                st.session_state["messages"] = []

            st.success(f"Processed '{uploaded_file.name}' into {num_chunks} chunks!")

    if "processed_file" in st.session_state:
        st.success(f"📄 {st.session_state['processed_file']}")
        st.caption(f"{st.session_state['num_chunks']} chunks indexed")

        if st.button("Clear & Upload New"):
            clear_collection()
            for key in ["processed_file", "num_chunks", "messages"]:
                st.session_state.pop(key, None)
            st.rerun()

# --- Main Chat Area ---
if "processed_file" not in st.session_state:
    st.info("👈 Upload a PDF in the sidebar to get started.")
else:
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 View Sources"):
                    for s in msg["sources"]:
                        st.caption(
                            f"**{s['source']}**, Page {s['page_number']} "
                            f"(relevance: {1 - s['distance']:.0%})"
                        )
                        st.text(s["text"][:300] + "...")
                        st.divider()

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(question)

            st.markdown(result["answer"])

            with st.expander("📚 View Sources"):
                for s in result["sources"]:
                    st.caption(
                        f"**{s['source']}**, Page {s['page_number']} "
                        f"(relevance: {1 - s['distance']:.0%})"
                    )
                    st.text(s["text"][:300] + "...")
                    st.divider()

        # Save to history
        st.session_state["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })