import os
import openai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # API-nyckeln från .env

def get_embedding(text: str):
    try:
        response = openai.embeddings.create(import os
import logging
from functools import lru_cache
from typing import List, Tuple, Dict, Any

import openai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type
)

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
    """
    Return the embedding vector for a given text, with automatic retries and in-memory caching.
    """
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
    """
    Compute cosine similarities between the question and each chunk, returning the top_k contexts.

    Returns:
        context: Combined text of the top chunks, separated by delimiters.
        top_chunks: List of tuples (score, chunk_text).
    """
    query_embed = get_embedding(question)

    similarities: List[Tuple[float, str]] = []
    for item in embedded_chunks:
        score = cosine_similarity(
            [query_embed],
            [item["embedding"]]
        )[0][0]
        similarities.append((score, item.get("text", "")))

    # Sort descending by score and take top_k
    top_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
    context = "\n---\n".join(chunk for _, chunk in top_chunks)
    logger.info(f"Selected top {top_k} chunks for the query")
    return context, top_chunks


def generate_gpt_answer(
    question: str,
    context: str,
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 700
) -> str:
    """
    Generate an answer from the GPT model given a question and supporting context.

    Raises:
        RuntimeError: If the OpenAI API returns an error.
    """
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
    # Exempel på hur du kan använda funktionerna
    sample_chunks = [
        {"text": "Example chunk A", "embedding": get_embedding("Example chunk A\")},
        {"text": "Example chunk B", "embedding": get_embedding("Example chunk B\")},
    ]
    question = "Vilka är de tre största marknaderna?"
    context, top = search_relevant_chunks(question, sample_chunks)
    print("Context for generation:\n", context)
    print("Svar från GPT:\n", generate_gpt_answer(question, context))

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
