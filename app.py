import os
import pickle
import hashlib
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from core.gpt_logic import search_relevant_chunks, generate_gpt_answer, get_embedding
import openai

# 🌍 Ladda API-nycklar etc.
load_dotenv()

# 🔐 OpenAI används i core/gpt_logic.py
def get_embedding_cache_name(source_id: str) -> str:
    hashed = hashlib.md5(source_id.encode("utf-8")).hexdigest()
    return os.path.join("embeddings", f"embeddings_{hashed}.pkl")

def save_embeddings(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_embeddings_if_exists(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

# 📥 Extrahera text
import pdfplumber  # Se till att det finns i requirements.txt också

def extract_text_from_file(file):
    text_output = ""

    if file.name.endswith(".pdf"):
        # 1. Läs text med fitz (PyMuPDF)
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text_output += "\n".join([page.get_text() for page in doc])

        # 2. Extrahera tabeller separat med pdfplumber
        file.seek(0)  # Viktigt: återställ filpekaren!
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row and any(cell for cell in row if cell):  # Inte tom rad
                                # Formatera raden som tabelltext
                                clean_row = "\t".join(cell.strip() if cell else "" for cell in row)
                                text_output += "\n" + clean_row
        except Exception as e:
            text_output += f"\n[⚠️ Kunde inte läsa tabeller med pdfplumber: {e}]"

    elif file.name.endswith(".html"):
        soup = BeautifulSoup(file.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text_output = soup.get_text(separator="\n")

    elif file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
        text_output = df.to_string(index=False)

    return text_output


@st.cache_data(show_spinner=False)
def fetch_html_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        tables = soup.find_all("table")
        table_texts = []
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                line = "\t".join(cell.get_text(strip=True) for cell in cells)
                if line:
                    table_texts.append(line)
        body_text = soup.get_text(separator="\n")
        clean_lines = [line.strip() for line in body_text.splitlines() if line.strip()]
        cleaned_text = "\n".join(clean_lines)
        return cleaned_text + "\n\n[TABELLINNEHÅLL]\n" + "\n".join(table_texts)
    except:
        return None

def chunk_text(text, chunk_size=1000, overlap=200):
    lines = text.split("\n")
    chunks, current, total_length = [], [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        current.append(line)
        total_length += len(line)
        if total_length >= chunk_size:
            chunks.append("\n".join(current))
            current, total_length = [], 0
    if current:
        chunks.append("\n".join(current))
    return chunks

# 🔍 Mönster för nyckeltal
import re
def is_key_figure(row):
    patterns = [
        r"\b\d+[\.,]?\d*\s*(SEK|MSEK|kr|miljoner|tkr|USD|\$|€|%)",
        r"(resultat|omsättning|utdelning|kassaflöde|kapital|intäkter|EBITDA|vinst).*?\d"
    ]
    return any(re.search(p, row, re.IGNORECASE) for p in patterns)

# 🌐 UI
st.set_page_config(page_title="📊 AI Rapportanalys", layout="wide")
st.markdown("<h1 style='color:#3EA6FF;'>📊 AI-baserad Rapportanalys</h1>", unsafe_allow_html=True)
st.image("https://www.appypie.com/dharam_design/wp-content/uploads/2025/05/headd.svg", width=120)

html_link = st.text_input("🌐 Rapport-länk (HTML)")
uploaded_file = st.file_uploader("📎 Ladda upp HTML, PDF eller Excel-fil", type=["html", "pdf", "xlsx", "xls"])

preview = ""
if uploaded_file:
    preview = extract_text_from_file(uploaded_file)
elif html_link:
    st.info("🔍 Hämtar innehåll...")
    preview = fetch_html_text(html_link)
else:
    preview = st.text_area("✏️ Klistra in text manuellt här:", "", height=200)

if preview:
    st.text_area("📄 Förhandsvisning:", preview[:5000], height=200)
else:
    st.warning("❌ Ingen text att analysera än.")

if "user_question" not in st.session_state:
    st.session_state.user_question = "Vilken utdelning per aktie föreslås?"

st.text_input("Fråga:", key="user_question")
user_question = st.session_state.user_question

if preview and len(preview.strip()) > 20:
    if st.button("🔍 Analysera"):
        with st.spinner("🔎 GPT analyserar..."):
            source_id = html_link or uploaded_file.name if uploaded_file else preview[:50]
            cache_file = get_embedding_cache_name(source_id)
            embedded_chunks = load_embeddings_if_exists(cache_file)

            if not embedded_chunks:
                chunks = chunk_text(preview)
                embedded_chunks = []
                for i, chunk in enumerate(chunks, 1):
                    try:
                        st.write(f"🔹 Chunk {i} – {len(chunk)} tecken")
                        embedding = get_embedding(chunk)
                        embedded_chunks.append({"text": chunk, "embedding": embedding})
                    except Exception as e:
                        st.error(f"❌ Fel vid embedding av chunk {i}: {e}")
                        st.stop()
                save_embeddings(cache_file, embedded_chunks)

            context, top_chunks = search_relevant_chunks(user_question, embedded_chunks)
            answer = generate_gpt_answer(user_question, context)

            st.success("✅ Svar klart!")
            st.markdown(f"### 🤖 GPT-4o svar:\n{answer}")

            key_figures = [row for row in answer.split("\n") if is_key_figure(row)]
            if key_figures:
                st.markdown("### 📊 Möjliga nyckeltal i svaret:")
                for row in key_figures:
                    st.markdown(f"- {row}")

            with st.expander("📚 Visa GPT-kontext"):
                for i, chunk in enumerate(top_chunks, 1):
                    st.markdown(f"**Chunk {i}:**\n{chunk[1]}")

            st.download_button("💾 Ladda ner svar (.txt)", answer, file_name="gpt_svar.txt")

            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in answer.split("\n"):
                pdf.multi_cell(0, 10, line)
            st.download_button("📄 Ladda ner svar (.pdf)", pdf.output(dest="S").encode("latin1"), file_name="gpt_svar.pdf")
else:
    st.warning("📝 Vänligen ange text, länk eller ladda upp en fil för att börja.")
