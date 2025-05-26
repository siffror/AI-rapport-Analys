import os
import pickle
import hashlib
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from core.gpt_logic import search_relevant_chunks, generate_gpt_answer, get_embedding, chunk_text, full_rapportanalys
from utils import extract_noterade_bolag_table
from ocr_utils import extract_text_from_image_or_pdf
import pdfplumber

# --- Load Environment Variables ---
# Load environment variables from a `.env` file. This is used to securely store sensitive data (e.g., API keys).
load_dotenv()

# --- Embedding Cache Functions ---
# These functions manage caching for embeddings generated from text.
# Caching helps avoid recomputation, speeding up operations and reducing API usage.

def get_embedding_cache_name(source_id: str) -> str:
    """
    Generate a unique cache file name for embeddings based on a hashed source ID.
    """
    hashed = hashlib.md5(source_id.encode("utf-8")).hexdigest()
    return os.path.join("embeddings", f"embeddings_{hashed}.pkl")

def save_embeddings(filename, data):
    """
    Save embeddings to a file for future reuse.
    Creates the directory if it does not exist.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_embeddings_if_exists(filename):
    """
    Load embeddings from a cache file if it exists.
    Returns None if the file does not exist.
    """
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

# --- File Text Extraction Functions ---
# Functions to extract text content from various file types such as PDFs, HTML, Excel sheets, and images.

def extract_text_from_file(file):
    """
    Extract text from a file based on its type.
    Supports PDF, HTML, and Excel formats.
    """
    text_output = ""
    if file.name.endswith(".pdf"):
        file.seek(0)
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_output += page_text + "\n"
        except Exception as e:
            st.warning(f"⚠️ Could not read PDF: {e}")

    elif file.name.endswith(".html"):
        soup = BeautifulSoup(file.read(), "html.parser")
        # Remove unnecessary tags to clean up the extracted text
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text_output = soup.get_text(separator="\n")

    elif file.name.endswith((".xlsx", ".xls")):
        # Read Excel files into a DataFrame and convert to text
        df = pd.read_excel(file)
        text_output = df.to_string(index=False)

    return text_output

# --- Fetch HTML Text from URL ---
# This function retrieves and cleans the textual content of a webpage.

@st.cache_data(show_spinner=False)
def fetch_html_text(url):
    """
    Fetch and clean textual content from a given URL.
    Removes unnecessary HTML elements like scripts and styles.
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        body_text = soup.get_text(separator="\n")
        clean_lines = [line.strip() for line in body_text.splitlines() if line.strip()]
        return "\n".join(clean_lines)
    except Exception as e:
        st.error(f"❌ Error fetching HTML: {e}")
        return ""

# --- Helper Function: Identify Key Figures ---
# This function identifies rows of text that likely contain financial key figures.

def is_key_figure(row):
    """
    Determine if a given text row contains financial key figures.
    Uses regular expressions to identify patterns such as currency values or financial terms.
    """
    patterns = [
        r"\b\d+[\.,]?\d*\s*(SEK|MSEK|kr|miljoner|tkr|USD|\$|€|%)",
        r"(resultat|omsättning|utdelning|kassaflöde|kapital|intäkter|EBITDA|vinst).*?\d"
    ]
    return any(re.search(p, row, re.IGNORECASE) for p in patterns)

# --- Streamlit UI Setup ---
# Configure the appearance and layout of the Streamlit app.

st.set_page_config(page_title="📊 AI Rapportanalys", layout="wide")
st.markdown("<h1 style='color:#3EA6FF;'>📊 AI-baserad Rapportanalys</h1>", unsafe_allow_html=True)
st.image("https://www.appypie.com/dharam_design/wp-content/uploads/2025/05/headd.svg", width=120)

# --- Input Fields for User Data ---
# Allow users to either upload a file or provide a URL for analysis.

html_link = st.text_input("🌐 Report Link (HTML)")
uploaded_file = st.file_uploader("📎 Upload HTML, PDF, Excel, or Image", type=["html", "pdf", "xlsx", "xls", "png", "jpg", "jpeg"])

# --- Text Extraction and Preview ---
# Extract and display a preview of the text content from the uploaded file or URL.

preview, ocr_text = "", ""
if uploaded_file:
    if uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
        # OCR for image files
        ocr_text, _ = extract_text_from_image_or_pdf(uploaded_file)
        st.text_area("📄 OCR Extracted Text:", ocr_text[:2000], height=200)
    else:
        preview = extract_text_from_file(uploaded_file)
elif html_link:
    preview = fetch_html_text(html_link)
else:
    preview = st.text_area("✏️ Paste text manually here:", "", height=200)

text_to_analyze = preview or ocr_text

# --- Text Preview Section ---
# Display a preview of the extracted text or a warning if no text is available.

if preview:
    st.text_area("📄 Preview:", preview[:5000], height=200)
else:
    st.warning("❌ No text available for analysis yet.")

# --- Full Analysis Button ---
# Perform a full report analysis using AI when the button is clicked.

if st.button("🔍 Full Report Analysis"):
    if text_to_analyze:
        with st.spinner("📊 GPT is analyzing the full report..."):
            st.markdown("### 🧾 Full AI Analysis:")
            st.markdown(full_rapportanalys(text_to_analyze))
    else:
        st.error("No text available for analysis.")

# --- Question and GPT Analysis ---
# Provide a text input for user questions and use GPT to analyze the extracted text.

if "user_question" not in st.session_state:
    st.session_state.user_question = "What dividend per share is proposed?"
st.text_input("Question:", key="user_question")

if text_to_analyze and len(text_to_analyze.strip()) > 20:
    if st.button("🔍 Analyze with GPT"):
        with st.spinner("🤖 GPT is analyzing..."):
            source_id = (html_link or uploaded_file.name if uploaded_file else text_to_analyze[:50]) + "-v2"
            cache_file = get_embedding_cache_name(source_id)
            embedded_chunks = load_embeddings_if_exists(cache_file)

            if not embedded_chunks:
                chunks = chunk_text(text_to_analyze)
                embedded_chunks = []
                for i, chunk in enumerate(chunks, 1):
                    st.write(f"🔹 Chunk {i} – {len(chunk)} characters")
                    try:
                        embedding = get_embedding(chunk)
                        embedded_chunks.append({"text": chunk, "embedding": embedding})
                    except Exception as e:
                        st.error(f"❌ Error embedding chunk {i}: {e}")
                        st.stop()
                save_embeddings(cache_file, embedded_chunks)

            context, top_chunks = search_relevant_chunks(st.session_state.user_question, embedded_chunks)
            st.code(context[:1000], language="text")
            answer = generate_gpt_answer(st.session_state.user_question, context)

            st.success("✅ Answer ready!")
            st.markdown(f"### 🤖 GPT-4o Answer:\n{answer}")

            key_figures = [row for row in answer.split("\n") if is_key_figure(row)]
            if key_figures:
                st.markdown("### 📊 Potential Key Figures in the Answer:")
                for row in key_figures:
                    st.markdown(f"- {row}")

            st.download_button("💾 Download Answer (.txt)", answer, file_name="gpt_answer.txt")
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in answer.split("\n"):
                pdf.multi_cell(0, 10, line)
            st.download_button("📄 Download Answer (.pdf)", pdf.output(dest="S").encode("latin1"), file_name="gpt_answer.pdf")
else:
    st.info("📝 Provide text, a link, or upload a file/image to begin.")
