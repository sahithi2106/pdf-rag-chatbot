import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import os

# =========================
# LOAD API KEY
# =========================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=api_key)

# =========================
# LOAD PDF
# =========================

print("Loading PDF...")

reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("PDF Loaded Successfully!")
print("Characters Extracted:", len(text))

# =========================
# CHUNK TEXT
# =========================

chunk_size = 300

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

print("Total Chunks:", len(chunks))

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("Loading Embedding Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded!")

# =========================
# CREATE EMBEDDINGS
# =========================

print("Creating Embeddings...")

embeddings = model.encode(chunks)

print("Embeddings Created!")

# =========================
# CREATE CHROMADB
# =========================

client = chromadb.Client()

try:
    client.delete_collection("resume")
except:
    pass

collection = client.create_collection("resume")

# =========================
# STORE CHUNKS
# =========================

for i, chunk in enumerate(chunks):
    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embeddings[i].tolist()]
    )

print("Resume Indexed Successfully!")

# =========================
# ASK QUESTION
# =========================

question = input("\nAsk a question: ")

# =========================
# RETRIEVE RELEVANT CHUNKS
# =========================

query_embedding = model.encode(question)

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)

context = "\n".join(results["documents"][0])

# =========================
# GEMINI
# =========================

model_gemini = genai.GenerativeModel("gemini-2.5-flash")

prompt = f"""
Answer the user's question ONLY using the context below.

Context:
{context}

Question:
{question}
"""

response = model_gemini.generate_content(prompt)

# =========================
# OUTPUT
# =========================

print("\nAnswer:\n")
print(response.text)