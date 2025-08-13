import json
import os
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Choose Gemini model
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

# Load document chunks
with open("parsed_chunks.json", "r") as f:
    chunks = json.load(f)

# Compute embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_texts = [c["text"] if "text" in c else c["content"] for c in chunks]
embeddings = embed_model.encode(chunk_texts)

# Build FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Retrieve relevant chunks
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = embed_model.encode([query])
    scores, indices = index.search(np.array(query_embedding), top_k)
    return [chunk_texts[i] for i in indices[0]]

# Answer query using Gemini
def answer_query(query):
    top_chunks = retrieve_relevant_chunks(query)
    context = "\n\n".join(top_chunks)

    prompt = f"""
You are an assistant for insurance policy document interpretation.

Based on the following clauses:
{context}

Analyze this query:
"{query}"

Respond STRICTLY in one short sentence, starting with 'Yes,' or 'No,' followed by the reason.
Do not include anything else.
"""

    response = model.generate_content(prompt)
    return response.text.strip()

# Example usage
if __name__ == "__main__":
    query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    result = answer_query(query)
    print(result)