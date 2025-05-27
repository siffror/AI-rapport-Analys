import os
import logging
from functools import lru_cache
from typing import List, Tuple, Dict, Any

from openai import OpenAI, OpenAIError
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

# Load environment variables
import streamlit as st

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    retry=retry_if_exception_type(OpenAIError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6)
)
@lru_cache(maxsize=512)
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    if not text:
        raise ValueError("Text för embedding får inte vara tom.")
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def search_relevant_chunks(
    question: str,
    embedded_chunks: List[Dict[str, Any]],
    top_k: int = 7
) -> Tuple[str, List[Tuple[float, str]]]:
    query_embed = get_embedding(question)
    question_words = set(question.lower().split())
    similarities = []

    for item in embedded_chunks:
        text = item.get("text", "")
        text_lower = text.lower()
        score = cosine_similarity([query_embed], [item["embedding"]])[0][0]

        # Fuzzy bonus: ge lite poäng för ord som matchar frågan
        fuzzy_bonus = sum(1 for word in question_words if word in text_lower) * 0.005
        score += fuzzy_bonus

        similarities.append((score, text))

    top_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
    context = "\n---\n".join([chunk for _, chunk in top_chunks])
    logger.info(f"Valde top {top_k} chunks för frågan.")
    return context, top_chunks

def generate_gpt_answer(
    question: str,
    context: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 700
) -> str:
    if not context.strip():
        raise ValueError("Kontext får inte vara tom vid generering.")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "Du är en AI som analyserar årsrapporter från företag. "
                    "Besvara användarens fråga baserat enbart på den kontext du får. "
                    "Var så specifik som möjligt, och inkludera nyckeltal och citat om det finns."
                )},
                {"role": "user", "content": f"Kontext:\n{context}\n\nFråga: {question}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        logger.error(f"OpenAI API-fel: {e}")
        raise RuntimeError(f"❌ Fel vid generering av svar: {e}")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Dela upp text i överlappande delar (chunks) för att möjliggöra effektiv embedding.
    """
    lines = text.split("\n")
    chunks = []
    current_chunk = []
    total_length = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        current_chunk.append(line)
        total_length += len(line)
        if total_length >= chunk_size:
            chunks.append("\n".join(current_chunk))
            overlap_text = "\n".join(current_chunk[-(overlap // 80):])
            current_chunk = [overlap_text] if overlap else []
            total_length = len(overlap_text)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def full_rapportanalys(text: str) -> str:
    system_prompt = (
        "Du är en ekonomisk AI-expert. Analysera årsrapporter och extrahera så mycket relevant information som möjligt. "
        "Fokusera på utdelning, omsättning, resultat, tillgångar, skulder, kassaflöde, vinst, viktiga händelser och eventuella risker. "
        "Strukturera svaret i tydliga sektioner med rubriker. Behåll samma språk som texten du får."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Fel vid analys: {e}"
