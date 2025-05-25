import os
import pickle
import hashlib
import requests
import re
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from core.gpt_logic import search_relevant_chunks, generate_gpt_answer, get_embedding
from utils import extract_noterade_bolag_table
from ocr_utils import extract_text_from_image_or_pdf
import pdfplumber

# 🌍 Ladda API-nycklar
load_dotenv()

# 🔐 Caching och sparning av embeddings
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

# 📄 Extrahera text från olika filtyper
def extract_text_from_file(file):
    text_output = ""

    if file.name.endswith(".pdf"):
        file.seek(0)
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_output += page_text + "\n"
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            clean_row = "\t".join(cell.strip() if cell else "" for cell in row)
                            text_output += clean_row + "\n"
        except Exception as e:
            text_output += f"\n[⚠️ Kunde inte läsa PDF med pdfplumber: {e}]"

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
uploaded_file = st.file_uploader("📎 Ladda upp HTML, PDF, Excel eller bild", type=["html", "pdf", "xlsx", "xls", "png", "jpg", "jpeg"])

preview = ""
ocr_text = ""

# 📥 Filhantering
if uploaded_file:
    if uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
        ocr_text, _ = extract_text_from_image_or_pdf(uploaded_file)
        st.text_area("📄 OCR-utläst text från bild:", ocr_text[:2000], height=200)

        if st.button("🔍 Analysera bildtext med GPT"):
            gpt_prompt = (
                "Här är en tabell hämtad från en bild av en finansiell rapport.\n"
                "Räkna hur många noterade bolag som listas:\n\n"
                f"{ocr_text}"
            )
            answer = generate_gpt_answer("Hur många noterade bolag listas?", gpt_prompt)
            st.markdown("### 🤖 GPT-4o svar:")
            st.write(answer)

    elif uploaded_file.name.endswith((".pdf", ".html", ".xlsx", ".xls")):
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

# 🧠 Fråga och GPT-svar
if "user_question" not in st.session_state:
    st.session_state.user_question = "Vilken utdelning per aktie föreslås?"

st.text_input("Fråga:", key="user_question")
user_question = st.session_state.user_question

if preview and len(preview.strip()) > 20:
    if st.button("🔍 Analysera text"):
        with st.spinner("🔎 GPT analyserar..."):
            if html_link:
                source_id = html_link
            elif uploaded_file:
                source_id = uploaded_file.name
            else:
                source_id = preview[:50]

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

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in answer.split("\n"):
                pdf.multi_cell(0, 10, line)
            st.download_button("📄 Ladda ner svar (.pdf)", pdf.output(dest="S").encode("latin1"), file_name="gpt_svar.pdf")
else:
    st.info("📝 Ange text, länk eller ladda upp en fil för att börja.")
