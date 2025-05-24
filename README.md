# 📊 AI-Powered Financial Report Analyzer

This Streamlit application uses **GPT-4o** from OpenAI to analyze financial reports (HTML, PDF, or plain text). Just upload a report, ask a question, and get a contextual GPT answer — powered by **RAG** (Retrieval-Augmented Generation).

## 🚀 Features

- 🔍 Analyze annual/quarterly reports in any language
- 🧠 GPT-4o answers based only on the report content
- 📄 Supports HTML, PDF, and pasted text
- 💾 Embeddings are generated once and cached
- 📤 Export answers as `.txt` or `.pdf`
- 🔐 Secrets are safely stored using Streamlit Cloud

## 🌐 Live App

👉 [Launch the app](https://ai-rapport-analys-ds24.streamlit.app/)

## ⚙️ Technologies

- Python 3.10+
- [Streamlit](https://streamlit.io)
- [OpenAI API](https://platform.openai.com/)
- PyMuPDF, BeautifulSoup, Scikit-learn, dotenv, and more

## 🧠 How It Works

1. Extracts text from uploaded or linked reports
2. Splits text into overlapping chunks
3. Creates OpenAI Embeddings per chunk
4. Finds top relevant chunks based on your question
5. Feeds those chunks + your question to GPT-4o

## 🔐 API Key Setup

Add your key securely via Streamlit Cloud:

```toml
OPENAI_API_KEY = "..."
```
[More Info](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)


📦 ai-report-analyzer/
├── app.py
├── core/
│   └── gpt_logic.py
├── requirements.txt
├── .streamlit/
│   └── config.toml (optional)
└── README.md

✨ Created by
Developed by @Siffror Zakaria
as part of a data science/DS24 education project.


---

✅ Just replace `@yourusername` and `https://your-image-url.com` with your actual GitHub username and (optional) app screenshot.

Let me know if you want help uploading the screenshot to GitHub or linking your name.
