import sys
import os
import datetime
import pickle
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
from core.gpt_logic import search_relevant_chunks, generate_gpt_answer, get_embedding
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

import hashlib
import pickle

def get_embedding_cache_name(source_id: str) -> str:
    hashed = hashlib.md5(source_id.encode("utf-8")).hexdigest()
    return os.path.join("embeddings", f"embeddings_{hashed}.pkl")

def save_embeddings(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 👈 Skapa mappen automatiskt
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_embeddings_if_exists(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

# 🛠 Lägg till projektets rotmapp i sökvägen så "core" hittas

load_dotenv()

# 📄 Extrahera text från uppladdad fil (PDF eller HTML)
def extract_text_from_file(file):
    import fitz  # PyMuPDF
    from bs4 import BeautifulSoup

    if file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])

    elif file.name.endswith(".html"):
        soup = BeautifulSoup(file.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    return ""

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
        full_text = cleaned_text + "\n\n[TABELLINNEHÅLL]\n" + "\n".join(table_texts)

        return full_text[:20000]
    except:
        return None

def chunk_text(text, chunk_size=1000, overlap=200):
    lines = text.split("\n")
    chunks = []
    current = []
    total_length = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        current.append(line)
        total_length += len(line)

        if total_length >= chunk_size:
            chunks.append("\n".join(current))
            total_length = 0
            current = []

    if current:
        chunks.append("\n".join(current))

    return chunks

# 🌐 UI
st.set_page_config(page_title="📊 AI Rapportanalys", layout="wide")
st.markdown("<h1 style='color:#3EA6FF;'>📊 AI-baserad Rapportanalys</h1>", unsafe_allow_html=True)
st.image("https://www.appypie.com/dharam_design/wp-content/uploads/2025/05/headd.svg", width=120)

html_link = st.text_input("🌐 Rapport-länk (HTML)")
uploaded_file = st.file_uploader("📎 Eller ladda upp HTML- eller PDF-fil", type=["html", "pdf"])

preview = None

# 1. Om fil laddas upp
if uploaded_file:
    preview = extract_text_from_file(uploaded_file)

# 2. Om länk anges
elif html_link:
    st.info("🔍 Hämtar innehåll...")
    preview = fetch_html_text(html_link)

# 3. Om man vill klistra in text manuellt
else:
    preview = st.text_area("✏️ Klistra in text manuellt här:", "", height=200)


if preview:
    st.text_area("📄 Förhandsvisning:", preview[:5000], height=200)
else:
    st.warning("❌ Ingen text att analysera än.")

user_question = st.text_input("Fråga:", "Vilken utdelning per aktie föreslås?")


import re  # För nyckeltalsmönster

def is_key_figure(row):
    patterns = [
        r"\b\d+[\.,]?\d*\s*(SEK|MSEK|kr|miljoner|tkr|USD|\$|€|%)",  # Ex: 5,50 SEK
        r"(resultat|omsättning|utdelning|kassaflöde|kapital|intäkter|EBITDA|vinst).*?\d"  # Ex: resultat 12 miljoner
    ]
    return any(re.search(pat, row, re.IGNORECASE) for pat in patterns)


if preview and len(preview.strip()) > 20:  # visa knapp bara om tillräckligt med text finns
    if st.button("🔍 Analysera"):
        with st.spinner("🔎 GPT analyserar..."): 
            source_id = html_link if html_link else uploaded_file.name if uploaded_file else preview[:50]
            cache_file = get_embedding_cache_name(source_id)

            embedded_chunks = load_embeddings_if_exists(cache_file)

            if not embedded_chunks:
                chunks = chunk_text(preview)
                embedded_chunks = [{"text": chunk, "embedding": get_embedding(chunk)} for chunk in chunks]
                save_embeddings(cache_file, embedded_chunks)

            context, top_chunks = search_relevant_chunks(user_question, embedded_chunks)
            answer = generate_gpt_answer(user_question, context)

            st.success("✅ Svar klart!")
            st.markdown(f"### 🤖 GPT-4o svar:\n{answer}")

            possible_key_figures = [row for row in answer.split("\n") if is_key_figure(row)]
            if possible_key_figures:
                st.markdown("### 📊 Möjliga nyckeltal i svaret:")
                for row in possible_key_figures:
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
            pdf_bytes = pdf.output(dest="S").encode("latin1")
            st.download_button("📄 Ladda ner svar (.pdf)", pdf_bytes, file_name="gpt_svar.pdf")
else:
    st.warning("📝 Vänligen ange text, länk eller ladda upp en fil för att börja.")


