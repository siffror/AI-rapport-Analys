import os
import logging
from functools import lru_cache
from typing import List, Tuple, Dict, Any

import openai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    retry=retry_if_exception_type(openai.error.OpenAIError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6)
)
@lru_cache(maxsize=512)
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    if not text:
        raise ValueError("Text for embedding must not be empty.")
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    embedding = response.data[0].embedding
    logger.debug(f"Generated embedding for text of length {len(text)}")
    return embedding

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
    logger.info(f"Selected top {top_k} chunks for the query")
    return context, top_chunks

def generate_gpt_answer(
    question: str,
    context: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 700
) -> str:
    if not context.strip():
        raise ValueError("Context for generation must not be empty.")
    try:
        response = openai.chat.completions.create(
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
        answer = response.choices[0].message.content
        logger.debug("Generated GPT answer successfully")
        return answer
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise RuntimeError(f"❌ Fel vid generering av svar: {e}")

if __name__ == "__main__":
    # Enkel testkörning
    sample_chunks = [
        {"text": "Bolaget har haft stark tillväxt i USA.", "embedding": get_embedding("Bolaget har haft stark tillväxt i USA.")},
        {"text": "Europa stod för 45% av omsättningen under Q4.", "embedding": get_embedding("Europa stod för 45% av omsättningen under Q4.")},
        {"text": "Asien är en växande marknad enligt senaste rapporten.", "embedding": get_embedding("Asien är en växande marknad enligt senaste rapporten.")}
    ]
    question = "Vilka är de tre största marknaderna?"
    context, top = search_relevant_chunks(question, sample_chunks)
    print("Context for generation:\n", context)
    print("Svar från GPT:\n", generate_gpt_answer(question, context))
