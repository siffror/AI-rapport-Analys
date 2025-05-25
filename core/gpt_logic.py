import os
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Läs in API-nyckel från .env
load_dotenv()
client = OpenAI()  # ← DETTA RÄCKER om du har OPENAI_API_KEY i .env

def get_embedding(text: str):
    try:
        return client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding
    except Exception as e:
        raise RuntimeError(f"❌ Fel vid skapande av embedding: {e}")

def search_relevant_chunks(question, embedded_chunks, top_k=3):
    query_embed = get_embedding(question)
    similarities = [
        (cosine_similarity([query_embed], [item["embedding"]])[0][0], item["text"])
        for item in embedded_chunks
    ]
    top_chunks = sorted(similarities, reverse=True)[:top_k]
    context = "\n---\n".join([chunk for _, chunk in top_chunks])
    return context, top_chunks

def generate_gpt_answer(question: str, context: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Du är en AI som analyserar årsrapporter. "
                    "Besvara användarens fråga baserat på kontexten nedan. Var tydlig och konkret."
                )},
                {"role": "user", "content": f"{context}\n\nFråga: {question}"}
            ],
            temperature=0.3,
            max_tokens=700
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"❌ Fel vid generering av GPT-svar: {e}")
