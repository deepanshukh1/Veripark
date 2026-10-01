"""Sample target: a tiny FAQ chatbot API.  Run:  uvicorn mock_bot.app:app --port 8000"""
from fastapi import FastAPI
from pydantic import BaseModel

from .kb import answer

app = FastAPI(title="Marlow & Finch Bank FAQ bot (fictional, with planted errors)")


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    text, faq_id = answer(req.question)
    return {"answer": text, "matched_faq": faq_id}
