from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.gpt_logic import generate_gpt_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(item: Question):
    # Just enkel svar utan embeddings
    answer = generate_gpt_answer(item.question, context="")
    return {"answer": answer}
