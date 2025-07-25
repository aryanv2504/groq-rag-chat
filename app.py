import streamlit as st
import os
from dotenv import load_dotenv
from utils import load_pdf, split_text, create_embeddings, create_faiss_index
from rag_chain import retrieve_chunks, generate_answer

load_dotenv()  # Loads .env and GROQ_API_KEY

st.set_page_config(page_title="RAG Chat System with Groq", layout="wide")
st.sidebar.title("Document Upload")
uploaded_files = st.sidebar.file_uploader("Choose PDF files here", type=["pdf"], accept_multiple_files=True)
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200)
k = st.sidebar.slider("Top K Results", 1, 10, 4)
model_name = st.sidebar.selectbox("Groq Model", ["llama3-8b-8192", "mixtral-8x7b-32768"])
process_button = st.sidebar.button("Process PDFs", use_container_width=True)

st.title("RAG Chat System with Groq")
st.markdown("Upload PDFs and chat with your documents using Groq’s AI.")

if process_button and uploaded_files:
    with st.spinner("Processing documents..."):
        all_text = ""
        for file in uploaded_files:
            all_text += load_pdf(file) + "\n"
        chunks = split_text(all_text, chunk_size=chunk_size, overlap=chunk_overlap)
        model, embeddings, chunks = create_embeddings(chunks)
        index = create_faiss_index(embeddings)
        st.session_state["chunks"] = chunks
        st.session_state["model"] = model
        st.session_state["index"] = index
        st.success("Documents processed and embeddings created!")

if "index" in st.session_state and "model" in st.session_state and "chunks" in st.session_state:
    st.subheader("Chat with Documents")
    user_question = st.text_input("Your question:", placeholder="Type your question here...")
    if st.button("Ask") and user_question.strip():
        with st.spinner("Generating answer..."):
            context = retrieve_chunks(
                query=user_question,
                model=st.session_state["model"],
                index=st.session_state["index"],
                chunks=st.session_state["chunks"],
                top_k=k
            )
            answer = generate_answer(context, user_question, model_name=model_name)
            st.markdown("**Answer:**")
            st.write(answer)
else:
    st.info("Please upload and process PDFs to begin chatting.")

# Quick Actions and Status (optional, same as before)
