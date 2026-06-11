import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key from .env
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=api_key)

# Load model
model = genai.GenerativeModel("gemini-2.5-flash")

# Ask Gemini
response = model.generate_content("Explain RAG in one sentence.")

print("\nGemini Response:\n")
print(response.text)