import os
import openai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# 🔐 Ladda API-nyckel från miljövariabler (Streamlit Secrets eller .env)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 🧠 Generera embedding från en textsträng
# core/gpt_logic.py

def get_embedding(text: str):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        import traceback
        print("❌ Full traceback:", traceback.format_exc())  # Visar hela felet i loggen
        raise RuntimeError(f"❌ Fel vid skapande av embedding: {e}")


# 🔍 Sök relevanta chunkar baserat på frågan
def search_relevant_chunks(question: str, embedded_chunks: list, top_k: int = 3):
    query_embed = get_embedding(question)
    similarities = []

    for item in embedded_chunks:
        score = cosine_similarity([query_embed], [item["embedding"]])[0][0]
        similarities.append((score, item["text"]))

    top_chunks = sorted(similarities, reverse=True)[:top_k]
    context = "\n---\n".join([chunk for _, chunk in top_chunks])

    return context, top_chunks

# 🤖 Generera svar med GPT-4o
def generate_gpt_answer(question: str, context: str) -> str:
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Du är en AI som analyserar årsrapporter från företag. "
                        "Besvara användarens fråga baserat enbart på den kontext du får. "
                        "Var så specifik som möjligt, och om siffror, nyckeltal eller direkta formuleringar från texten finns – inkludera dem tydligt. "
                        "Svar ska vara konkreta, gärna med punktlistor eller direkta utdrag från rapporten där det är relevant."
                    )
                },
                {
                    "role": "user",
                    "content": f"Använd nedanstående information för att besvara frågan:\n{context}\n\nFråga: {question}"
                }
            ],
            temperature=0.3,
            max_tokens=700
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"❌ GPT-förfrågan misslyckades: {e}")
