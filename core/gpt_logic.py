import os
import logging
from functools import lru_cache
from typing import List, Tuple, Dict, Any

from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

# Load environment variables
load_dotenv()
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=None  # Undvik att något extern config skickas med
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
    top_k: int = 3
) -> Tuple[str, List[Tuple[float, str]]]:
    query_embed = get_embedding(question)
    similarities = []
    for item in embedded_chunks:
        score = cosine_similarity([query_embed], [item["embedding"]])[0][0]
        similarities.append((score, item.get("text", "")))
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
