# Phoenix RAG

## Introduction
This is a RAG implementation using LLamaIndex, FastAPI and Streamlit for the Specific Queries related to the Company Law. It allows you to prepare a RAG index from a folder of documents and use it to answer questions. Also you can be able to chat with the lawyer agent who will answer your queries.

## Installation

```bash
pip install -r requirements.txt
```

## Run the server

```bash
uvicorn server:app --reload
```

## Run the app

```bash
streamlit run app.py
```
