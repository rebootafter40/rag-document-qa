"""
app.py — Streamlit frontend for the RAG Document Q&A app.
"""
import streamlit as st
from src.retriever import ingest_pdf, retrieve
from src.qa_chain import ask
from src.vector_store import clear_collection, delete_document
from src.logging_config import setup_logging
import tempfile
import os

# Initialize logging for all src modules
setup_logging()


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

# Initialize document list if it doesn't exist
    if "processed_files" not in st.session_state:
        st.session_state["processed_files"] = []

    if uploaded_file:
        # Check for duplicate
        already_uploaded = any(
            f["name"] == uploaded_file.name
            for f in st.session_state["processed_files"]
        )

        if already_uploaded:
            st.warning(f"'{uploaded_file.name}' is already uploaded.")
        elif st.button("Process Document", type="primary"):
            with st.spinner("Processing document... This may take a minute."):
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    num_chunks = ingest_pdf(tmp_path, original_filename=uploaded_file.name)

                    st.session_state["processed_files"].append({
                        "name": uploaded_file.name,
                        "num_chunks": num_chunks,
                    })

                    if "messages" not in st.session_state:
                        st.session_state["messages"] = []

                    st.success(f"Processed '{uploaded_file.name}' into {num_chunks} chunks!")

                except ValueError as e:
                    st.error(f"⚠️ {e}")
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred while processing: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    # Show uploaded documents
    if st.session_state.get("processed_files"):
        st.divider()
        st.subheader("📚 Uploaded Documents")
        for doc in st.session_state["processed_files"]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📄 {doc['name']} — {doc['num_chunks']} chunks")
            with col2:
                if st.button("✕", key=f"remove_{doc['name']}"):
                    delete_document(doc["name"])
                    st.session_state["processed_files"] = [
                        f for f in st.session_state["processed_files"]
                        if f["name"] != doc["name"]
                    ]
                    st.rerun()

        if st.button("Clear All Documents"):
            clear_collection()
            st.session_state["processed_files"] = []
            st.session_state.pop("messages", None)
            st.rerun()

    # --- Retrieval Settings ---
    st.divider()
    st.header("⚙️ Settings")
    use_reranking = st.toggle(
        "Enable reranking",
        value=False,
        help="Uses a cross-encoder model to improve retrieval accuracy. "
             "Slightly slower but can give better answers.",
    )


# --- Main Chat Area ---
if not st.session_state.get("processed_files"):
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
                try:
                    result = ask(question, use_reranking=use_reranking, conversation_history=st.session_state.get("messages", [])[:-1])
                except (ValueError, RuntimeError) as e:
                    # Show API or validation errors inline in the chat
                    result = {"answer": f"⚠️ {e}", "sources": []}
                except Exception as e:
                    result = {"answer": f"❌ Something went wrong: {e}", "sources": []}

            st.markdown(result["answer"])

            # Only show sources expander if there are sources
            if result["sources"]:
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