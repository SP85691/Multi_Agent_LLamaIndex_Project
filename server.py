from fastapi import FastAPI
from typing import Annotated, List, Optional
from pydantic import BaseModel, Field, HttpUrl, EmailStr
from process import Chatbot

app = FastAPI()

chatbot = Chatbot()

@app.get("/")
async def root():
    return {"message": "Welcome to the Phoenix RAG API"}

@app.post("/prepare-rag")
async def prepare_rag(folder_path: str):
    query_engine = chatbot.prepare_rag(folder_path=folder_path)
    return {"message": "RAG prepared successfully"}

@app.post("/ask")
async def ask(query: str):
    response = chatbot.ask(query)
    return {"message": response}
