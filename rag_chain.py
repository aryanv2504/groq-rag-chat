import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI  # This is the Groq-compatible OpenAI client!

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

def retrieve_chunks(query, model, index, chunks, top_k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return "\n".join([chunks[i] for i in I[0]])

def generate_answer(context, question, model_name="llama3-8b-8192"):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()
