import openai
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def retrieve_chunks(query, model, index, chunks, top_k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return "\n".join([chunks[i] for i in I[0]])

def generate_answer(context, question):
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()
