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

# --- Miljöinställningar ---
load_dotenv()

# --- Cache-funktioner ---
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

# --- Textutvinning ---
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

# --- Chunking & filtrering ---
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

# --- GPT: Full rapportanalys ---
def full_rapportanalys(text: str) -> str:
    system_prompt = (
        "Du är en erfaren finansiell analytiker med djup förståelse för företagsekonomi, strategi och rapportanalys. "
        "Du får en årsrapport eller annan finansiell text och ska göra en komplett analys av bolaget baserat på innehållet. "
        "Analysera kreativt, identifiera mönster, tolka siffror, och lyft fram både styrkor, svagheter, risker och möjligheter. "
        "Om något verkar saknas eller är oklart – kommentera det. Dra slutsatser där det är möjligt, men gissa aldrig. "
        "Ge en strukturerad analys med rubriker som: Översikt, Finansiell Sammanfattning, Väsentliga Händelser, Risker, Kommentarer. "
        "Svara på samma språk som texten du får, oavsett om det är svenska, engelska eller annat. "
        "Om användaren ställer en fråga på annat språk än rapporten – anpassa svaret till frågespråket, men citera från originaltexten där det är relevant."
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

# Kombinera extraherad text
text_to_analyze = preview or ocr_text

if preview:
    st.text_area("📄 Förhandsvisning:", preview[:5000], height=200)
else:
    st.warning("❌ Ingen text att analysera än.")

# 🔍 Fullständig rapportanalys
if st.button("🔍 Fullständig rapportanalys"):
    with st.spinner("📊 GPT analyserar hela rapporten..."):
        if text_to_analyze:
            full_summary = full_rapportanalys(text_to_analyze)
            st.markdown("### 🧾 Fullständig AI-analys:")
            st.markdown(full_summary)
            st.download_button("💾 Ladda ner fullständig analys (.txt)", full_summary, file_name="full_rapportanalys.txt")
        else:
            st.error("Ingen text tillgänglig för analys.")

# 🧠 Fråga och GPT-svar
if "user_question" not in st.session_state:
    st.session_state.user_question = "Vilken utdelning per aktie föreslås?"

st.text_input("Fråga:", key="user_question")
user_question = st.session_state.user_question

if text_to_analyze and len(text_to_analyze.strip()) > 20:
    if st.button("🔍 Analysera med GPT"):
        with st.spinner("🤖 GPT analyserar..."):
            source_id = html_link or (uploaded_file.name if uploaded_file else text_to_analyze[:50])
            cache_file = get_embedding_cache_name(source_id)
            embedded_chunks = load_embeddings_if_exists(cache_file)

            if not embedded_chunks:
                chunks = chunk_text(text_to_analyze)
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

            context = "\n---\n".join(chunk["text"] for chunk in embedded_chunks)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{context}\n\nFråga: {user_question}"}
            ]

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )

            answer = response.choices[0].message.content

            

            st.success("✅ Svar klart!")
            st.markdown("### 🤖 GPT-4o svar:")
            st.info(answer)

            # Filtrera nyckeltal och undvik dubbletter
            key_figures = list(set(
                row.strip() for row in answer.split("\n")
                if is_key_figure(row) and len(row.strip()) > 10
            ))
            if st.checkbox("🔍 Visa hela GPT-kontext (debug)"):
                st.text_area("🧠 GPT får denna text:", context[:10000], height=300)

            if "ingen specifik information om föreslagen utdelning" in answer.lower():
                st.warning("⚠️ GPT hittade ingen specifik information om föreslagen utdelning. Du kan behöva kontrollera årsrapportens senare delar eller separata utdelningsbesked.")
            elif key_figures:
                st.markdown("### 📊 Möjliga nyckeltal i svaret:")
                for row in key_figures:
                    st.markdown(f"- {row}")

        

            st.download_button("💾 Ladda ner svar (.txt)", answer, file_name="gpt_svar.txt")

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in answer.split("\n"):
                pdf.multi_cell(0, 10, line)
            st.download_button("📄 Ladda ner svar (.pdf)", pdf.output(dest="S").encode("latin1"), file_name="gpt_svar.pdf")
else:
    st.info("📝 Ange text, länk eller ladda upp en fil eller bild för att börja.")

