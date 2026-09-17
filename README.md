# Full Stack Academy RAG Chatbot (Streamlit)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your Groq key
```

Put your PDFs (what was `/content/fsa_pds` in Colab) into the `pdfs/` folder.

## Run
```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this folder to GitHub (secrets.toml is git-ignored).
2. Create an app at share.streamlit.io pointing to `app.py`.
3. In the app's **Settings → Secrets**, paste: `GROQ_API_KEY = "gsk_..."`
