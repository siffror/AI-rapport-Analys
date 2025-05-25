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
import openai

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
        r"\b\d+[\.,]?\d*\s*(SEK|MSEK|kr|miljoner|tkr|USD|\$|\u20ac|%)",
        r"(resultat|omsättning|utdelning|kassaflöde|kapital|intäkter|EBITDA|vinst).*?\d"
    ]
    return any(re.search(p, row, re.IGNORECASE) for p in patterns)

def full_rapportanalys(text: str) -> str:
    system_prompt = (
        "Du är en ekonomisk AI-expert. Analysera årsrapporter och extrahera så mycket relevant information som möjligt. "
        "Fokusera på utdelning, omsättning, resultat, tillgångar, skulder, kassaflöde, vinst, viktiga händelser och eventuella risker. "
        "Strukturera svaret i tydliga sektioner med rubriker. Behåll samma språk som texten du får."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Fel vid analys: {e}"

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
                "Här är en tabell hämtad från en bild av en finansiell rapport.
"
                "Räkna hur många noterade bolag som listas:

"
                f"{ocr_text}"
            )
            answer = generate_gpt_answer("Hur många noterade bolag listas?", gpt_prompt)
            st.markdown("### 🤖 GPT-4o svar:")
            st.write(answer)
        answer = generate_gpt_answer("Hur många noterade bolag listas?", gpt_prompt)
        st.markdown("### 🤖 GPT-4o svar:")
        st.write(answer)
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

# 🔍 Kombinera extraherad text från dokument eller bild
text_to_analyze = preview or ocr_text

# 🔍 Fullständig analysknapp sist:
if text_to_analyze and len(text_to_analyze.strip()) > 20:
    if st.button("\ud83d\udd0d Fullständig rapportanalys"):
        with st.spinner("\ud83d\udcca GPT analyserar hela rapporten..."):
            full_summary = full_rapportanalys(text_to_analyze)
            st.markdown("### \ud83e\uddfe Fullständig AI-analys:")
            st.markdown(full_summary)
