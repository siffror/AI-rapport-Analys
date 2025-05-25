import os
import openai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # API-nyckeln från .env

def get_embedding(text: str):
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"❌ Fel vid skapande av embedding: {e}")

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
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Du är en AI som analyserar årsrapporter från företag. "
                    "Besvara användarens fråga baserat enbart på den kontext du får. "
                    "Var så specifik som möjligt, och inkludera nyckeltal och citat om det finns."
                )},
                {"role": "user", "content": f"Kontext:\n{context}\n\nFråga: {question}"}
            ],
            temperature=0.3,
            max_tokens=700
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"❌ Fel vid generering av svar: {e}")
