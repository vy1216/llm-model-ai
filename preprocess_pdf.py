# preprocess_pdf.py
import json
from utils import extract_text_from_pdf, chunk_text

pdf_path = "sample.pdf"  # change this to your own

text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)

with open("parsed_chunks.json", "w", encoding="utf-8") as f:
    json.dump([{"text": c} for c in chunks], f, indent=2)

print(f"✅ Extracted {len(chunks)} chunks from {pdf_path}")