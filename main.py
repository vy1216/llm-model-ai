import streamlit as st
import fitz  # PyMuPDF
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
from PIL import Image
import pytesseract
import re

# Direct path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === Security config (added) ===
MAX_MAIN_UPLOAD_MB = 25  # PDFs
MAX_SIDEBAR_UPLOAD_MB = 10  # images / txt
ALLOWED_PDF_MIME = {"application/pdf"}
ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg"}
ALLOWED_TEXT_MIME = {"text/plain"}

def _file_too_large(uploaded_file, max_mb: int) -> bool:
    try:
        return uploaded_file.size > max_mb * 1024 * 1024
    except Exception:
        return False

def _sanitize_text(s: str) -> str:
    # remove control & non-printable chars; collapse excessive whitespace
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

SYSTEM_PROMPT = (
    "You are a safe, read-only document assistant. "
    "Never execute code, browse the internet, open links, or request secrets. "
    "Only answer based strictly on the provided content. "
    "If the answer is not in the content, say so clearly."
)

# === Setup ===
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    st.warning("⚠ Environment variable GEMINI_API_KEY is not set. Add it to your .env or secrets.")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("🪄📜 DocuPilot — Your AI for every 'What, Why, and How.")
st.markdown("Upload PDFs or use the sidebar tool to ask questions about text or images.")

# Persistent folder
RECENT_FOLDER = "recent_files"
os.makedirs(RECENT_FOLDER, exist_ok=True)

if "recent_files" not in st.session_state:
    st.session_state.recent_files = {}
    for fname in os.listdir(RECENT_FOLDER):
        if fname.lower().endswith(".pdf"):
            with open(os.path.join(RECENT_FOLDER, fname), "rb") as f:
                st.session_state.recent_files[fname] = f.read()

if "index" not in st.session_state:
    st.session_state.index = None
if "chunk_texts" not in st.session_state:
    st.session_state.chunk_texts = []

# === Helpers ===
def extract_chunks_from_pdf(file_bytes, chunk_size=500, overlap=50):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    full_text = _sanitize_text(full_text)
    chunks = []
    for i in range(0, len(full_text), chunk_size - overlap):
        piece = full_text[i:i + chunk_size]
        if piece.strip():
            chunks.append({"content": piece})
    return chunks

def build_faiss_index(chunks):
    texts = [c["content"] for c in chunks if c.get("content", "").strip()]
    if not texts:
        return None, []
    embeddings = embed_model.encode(texts)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, texts

def recognize_speech_and_answer():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Speak now...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            query = recognizer.recognize_google(audio, language="en-IN")
            st.write(f"**Recognized:** {query}")
            answer_query(query)
        except sr.WaitTimeoutError:
            st.warning("⏳ No speech detected.")
        except sr.UnknownValueError:
            st.error("❌ Could not understand the audio")
        except sr.RequestError as e:
            st.error(f"⚠ API Error: {e}")

def answer_query(query):
    if st.session_state.index is None:
        st.warning("Please upload or select a PDF first.")
        return
    query_embedding = embed_model.encode([query])
    scores, indices = st.session_state.index.search(np.array(query_embedding), 3)
    top_chunks = "\n".join(st.session_state.chunk_texts[i] for i in indices[0])

    prompt = f"""
    {SYSTEM_PROMPT}

    Based on the following extracted clauses from the document:
    {top_chunks}

    Question: "{_sanitize_text(query)}"
    
    Please provide a detailed, elaborate answer explaining clearly:
    - The reasoning behind the answer
    - The relevant clause content
    - Any conditions or exceptions
    """

    try:
        response = model.generate_content(prompt)
        detailed_answer = (response.text or "").strip()
        exact_match_text = st.session_state.chunk_texts[indices[0][0]].strip()

        st.markdown("### 🧠 Detailed Answer")
        st.write(detailed_answer if detailed_answer else "No answer text returned by the model.")

        st.markdown(f"**📌 Exact Matched Text:**")
        st.code(exact_match_text)

    except Exception as e:
        st.error(f"⚠️ API call failed: {e}")

# === File Upload ===
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    all_chunks = []
    for file in uploaded_files:
        if file.type not in ALLOWED_PDF_MIME:
            st.error(f"❌ '{file.name}' is not a PDF.")
            continue
        if _file_too_large(file, MAX_MAIN_UPLOAD_MB):
            st.error(f"❌ '{file.name}' exceeds {MAX_MAIN_UPLOAD_MB} MB.")
            continue

        file_bytes = file.read()
        st.session_state.recent_files[file.name] = file_bytes
        with open(os.path.join(RECENT_FOLDER, file.name), "wb") as f:
            f.write(file_bytes)
        chunks = extract_chunks_from_pdf(file_bytes)
        all_chunks.extend(chunks)

    st.session_state.index, st.session_state.chunk_texts = build_faiss_index(all_chunks)
    if st.session_state.index is not None:
        st.success(f"✅ Processed {len(uploaded_files)} file(s)")
    else:
        st.warning("No valid text extracted to index.")

# === Recent Files Dropdown ===
if st.session_state.recent_files:
    st.subheader("📂 Recent Files")
    recent_choice = st.selectbox(
        "Select a recent file to process",
        ["-- Select --"] + list(st.session_state.recent_files.keys())
    )
    if recent_choice != "-- Select --":
        file_bytes = st.session_state.recent_files[recent_choice]
        chunks = extract_chunks_from_pdf(file_bytes)
        st.session_state.index, st.session_state.chunk_texts = build_faiss_index(chunks)
        if st.session_state.index is not None:
            st.success(f"✅ Loaded {recent_choice}")
        else:
            st.warning(f"No extractable text found in {recent_choice}.")

# === Q&A Section ===
st.subheader("🔍 Ask a question")
question = st.text_input("Enter your question")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Get Answer"):
        if question:
            answer_query(question)
        else:
            st.info("Please enter a question.")
with col2:
    if st.button("🎤 Ask with Voice"):
        recognize_speech_and_answer()

# === Sidebar: Extra Tool for Text/Image Q&A ===
st.sidebar.header("🛠 Extra Tool: Ask from Text or Image")
input_text = st.sidebar.text_area("Paste your text here")

uploaded_text_file = st.sidebar.file_uploader("Or upload a text file", type=["txt"])
if uploaded_text_file:
    if uploaded_text_file.type in ALLOWED_TEXT_MIME and not _file_too_large(uploaded_text_file, MAX_SIDEBAR_UPLOAD_MB):
        input_text = _sanitize_text(uploaded_text_file.read().decode("utf-8", errors="ignore"))
    else:
        st.sidebar.error(f"Invalid or too-large text file (>{MAX_SIDEBAR_UPLOAD_MB} MB).")

uploaded_image = st.sidebar.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
if uploaded_image:
    if uploaded_image.type in ALLOWED_IMAGE_MIME and not _file_too_large(uploaded_image, MAX_SIDEBAR_UPLOAD_MB):
        try:
            img = Image.open(uploaded_image)
            ocr_text = pytesseract.image_to_string(img)
            ocr_text = _sanitize_text(ocr_text)
            st.sidebar.write("📜 Extracted Text from Image:")
            st.sidebar.code(ocr_text if ocr_text else "[No text detected]")
            input_text = (input_text + "\n" + ocr_text).strip()
        except Exception as e:
            st.sidebar.error(f"OCR failed: {e}")
    else:
        st.sidebar.error(f"Invalid or too-large image (>{MAX_SIDEBAR_UPLOAD_MB} MB).")

sidebar_question = st.sidebar.text_input("Your question about this text/image")
if st.sidebar.button("Ask Sidebar Tool"):
    if input_text.strip() and sidebar_question.strip():
        prompt = f"""
        {SYSTEM_PROMPT}

        Here is the provided text:
        {input_text}

        Question: "{_sanitize_text(sidebar_question)}"
        
        Provide a detailed and elaborate answer that is easy to understand.
        """
        try:
            response = model.generate_content(prompt)
            st.sidebar.markdown("### 🧠 Answer")
            st.sidebar.write((response.text or "").strip())
        except Exception as e:
            st.sidebar.error(f"⚠️ API call failed: {e}")
    else:
        st.sidebar.warning("Please provide text/image and a question.")
