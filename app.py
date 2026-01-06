import streamlit as st
from dotenv import load_dotenv
from utils import load_pdf, split_text, create_embeddings, create_faiss_index
from rag_chain import retrieve_chunks, generate_answer

load_dotenv()

st.set_page_config(page_title="RAG Chat System with Groq", layout="wide")

st.sidebar.title("Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200)
top_k = st.sidebar.slider("Top K Results", 1, 10, 4)

model_name = st.sidebar.selectbox(
    "Groq Model",
    ["llama3-8b-8192"]  # SAFE MODEL
)

process_button = st.sidebar.button("Process PDFs", use_container_width=True)

st.title("RAG Chat System with Groq")
st.markdown("Upload PDFs and chat with your documents using Groq LLMs.")

if process_button and uploaded_files:
    with st.spinner("Processing documents..."):
        full_text = ""
        for file in uploaded_files:
            full_text += load_pdf(file) + "\n"

        chunks = split_text(
            full_text,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )

        model, embeddings, chunks = create_embeddings(chunks)
        index = create_faiss_index(embeddings)

        st.session_state["chunks"] = chunks
        st.session_state["model"] = model
        st.session_state["index"] = index

        st.success("Documents processed successfully!")

if all(k in st.session_state for k in ["chunks", "model", "index"]):
    st.subheader("Chat with your documents")

    user_question = st.text_input(
        "Ask a question",
        placeholder="What is this document about?"
    )

    if st.button("Ask") and user_question.strip():
        with st.spinner("Generating answer..."):
            context = retrieve_chunks(
                query=user_question,
                model=st.session_state["model"],
                index=st.session_state["index"],
                chunks=st.session_state["chunks"],
                top_k=top_k
            )

            answer = generate_answer(
                context=context,
                question=user_question,
                model_name=model_name
            )

            st.markdown("### Answer")
            st.write(answer)
else:
    st.info("Upload and process PDFs to start chatting.")
