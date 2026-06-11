# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) application built using:

- Python
- ChromaDB
- Sentence Transformers
- Gemini API
- PyPDF

## Features

- Extract text from PDF documents
- Generate embeddings
- Store embeddings in ChromaDB
- Semantic similarity search
- Gemini-powered answer generation

## Example

Ask questions about a resume PDF such as:

- What is the educational background of the applicant?
- What technical skills does the applicant have?
- What projects has the applicant worked on?

## Installation

```bash
pip install chromadb sentence-transformers google-generativeai pypdf python-dotenv
```

## Run

```bash
python rag.py
```