# core/gpt_logic.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def search_relevant_chunks(question, embedded_chunks, top_k=3):
    query_embed = get_embedding(question)

    similarities = []
    for item in embedded_chunks:
        score = cosine_similarity([query_embed], [item["embedding"]])[0][0]
        similarities.append((score, item["text"]))

    top_chunks = sorted(similarities, reverse=True)[:top_k]
    context = "\n---\n".join([chunk for _, chunk in top_chunks])
    return context, top_chunks

def generate_gpt_answer(question: str, context: str):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
    "Du är en AI som analyserar årsrapporter från företag. "
    "Besvara användarens fråga baserat enbart på den kontext du får. "
    "Var så specifik som möjligt, och om siffror, nyckeltal eller direkta formuleringar från texten finns – inkludera dem tydligt. "
    "Svar ska vara konkreta, gärna med punktlistor eller direkta utdrag från rapporten där det är relevant."
)},


            {"role": "user", "content": f"Använd nedanstående information för att besvara frågan:\n{context}\n\nFråga: {question}"}
        ],
        temperature=0.3,
        max_tokens=700
    )
    return response.choices[0].message.content
