import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve_chunks(query, model, index, chunks, top_k=4):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    return "\n".join(retrieved)


def generate_answer(context, question, model_name):
    prompt = f"""
You are a helpful AI assistant.
Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content
