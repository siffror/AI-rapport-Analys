import os
import openai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text: str):
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        raise RuntimeError(f"❌ Fel vid skapande av embedding: {e}")

def generate_gpt_answer(question: str, context: str):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Du är en AI som analyserar årsrapporter från företag. "
                    "Besvara användarens fråga baserat enbart på den kontext du får. "
                    "Var så specifik som möjligt, och om siffror, nyckeltal eller direkta formuleringar från texten finns – inkludera dem tydligt. "
                    "Svar ska vara konkreta, gärna med punktlistor eller direkta utdrag från rapporten där det är relevant."
                )},
                {"role": "user", "content": f"{context}\n\nFråga: {question}"}
            ],
            temperature=0.3,
            max_tokens=700
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"❌ Fel vid generering av svar: {e}")
