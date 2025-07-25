# Groq RAG PDF Chatbot

An AI chatbot that enables you to upload PDFs and ask questions about their contents using Groq’s advanced LLM models and an intuitive Streamlit UI.

---

## 🚀 Live Demo

Experience the app instantly — no setup needed!  
**[Try the live demo here](https://groq-rag-chat-hihrbeksbgkpqsqrsuy54s.streamlit.app/)**

#### Link- https://groq-rag-chat-hihrbeksbgkpqsqrsuy54s.streamlit.app/
---

## Interface Preview
![RAG Chat System Interface](interface%20rag%20model.jpg)


---

## Features

- Multi-PDF upload and intelligent chunking  
- Semantic embeddings and efficient vector search (SentenceTransformers + FAISS)  
- Groq LLM-powered answer generation  
- Customizable chunk size, overlap, and top-K retrieval  
- Quick action buttons: Summaries, key points, FAQ  
- User-friendly Streamlit web interface  

---

## Getting Started

### Prerequisites

- Python 3.8 or later  
- Groq API key (stored in `.env` as `GROQ_API_KEY`)  
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/aryanv2504/groq-rag-chat-streamlit.git
   cd groq-rag-chat-streamlit
   ```

2. Create and activate a virtual environment:

   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Add your Groq API key to the `.env` file at the project root:

   ```
   GROQ_API_KEY=gsk_your_groq_api_key_here
   ```

---

## Usage

Start the Streamlit app:

```
streamlit run app.py
```

- Upload one or more PDF documents via the sidebar.  
- Adjust chunk size, overlap, and top-K parameters for retrieval customization.  
- Click **Process PDFs** to extract, chunk, embed, and index your documents.  
- Ask questions to get answers powered by Groq’s LLM models.  
- Use quick action buttons to generate summaries, key points, or FAQ-style answers instantly.

---

## Project Structure

```
├── app.py                # Streamlit UI and main logic
├── rag_chain.py          # Retrieval & generation using Groq API
├── utils.py              # PDF processing, chunking, embedding, indexing
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables including Groq API key
└── interface_rag_model.jpg  # Screenshot of the chat interface
```



