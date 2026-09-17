"""
Full Stack Academy RAG Chatbot — Streamlit version of Rag_June.ipynb

Run:  streamlit run app.py
"""
import hashlib
import os
import re
import tempfile
from pathlib import Path

import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
WEBSITE_URLS = [
    "https://fullstackacademy.in/",
    "https://fullstackacademy.in/about/",
]
PDF_DIR = Path(__file__).parent / "pdfs"          # replaces /content/fsa_pds
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "openai/gpt-oss-120b"
LOGO_URL = "https://fullstackacademy.in/wp-content/uploads/2023/10/fsa-new-logo.png"
HISTORY_TURNS = 6  # how many past messages to send to the LLM

st.set_page_config(page_title="FSA RAG Chatbot", page_icon="🎓", layout="centered")


# ----------------------------------------------------------------------------
# API key (replaces google.colab.userdata)
# ----------------------------------------------------------------------------
def get_api_key() -> str | None:
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:  # no secrets.toml present
        key = None
    return key or os.environ.get("GROQ_API_KEY")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def scrape_website(urls: tuple[str, ...]) -> list[dict]:
    """Scrape each page separately (fixes the soup/soup1 mix-up in the notebook)."""
    pages = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except requests.RequestException as e:
            pages.append({"url": url, "text": "", "error": str(e)})
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        pages.append({"url": url, "text": text, "error": None})
    return pages


def load_pdfs(uploaded: list[tuple[str, bytes]]) -> list[Document]:
    docs: list[Document] = []
    # PDFs shipped with the app
    for path in sorted(PDF_DIR.glob("*.pdf")):
        docs.extend(PyPDFLoader(str(path)).load())
    # PDFs uploaded through the sidebar
    for name, data in uploaded:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            for d in PyPDFLoader(tmp_path).load():
                d.metadata["source"] = name
                docs.append(d)
        finally:
            os.unlink(tmp_path)
    return docs


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def build_vector_store(cache_key: str, _pages: list[dict], _uploaded: list[tuple[str, bytes]]):
    """Rebuilt only when cache_key changes (i.e. when the source files change)."""
    documents = [
        Document(page_content=p["text"], metadata={"source": p["url"]})
        for p in _pages if p["text"]
    ]
    documents.extend(load_pdfs(_uploaded))
    if not documents:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"FSA_{cache_key[:16]}",  # unique name avoids stale duplicates
    )
    return store, len(chunks)


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and a follow-up question, rewrite the follow-up as a "
     "standalone question that can be understood without the history. "
     "Do NOT answer it. If it is already standalone, return it unchanged."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant for Full Stack Academy. "
     "Answer questions about Full Stack Academy using ONLY the context below. "
     "If something isn't in the context, say it isn't mentioned.\n"
     "For questions about this conversation itself (e.g. 'what did I ask before?'), "
     "use the chat history instead of the context.\n\n"
     "Full Stack Academy context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


def to_lc_history(messages: list[dict]) -> list:
    out = []
    for m in messages[-HISTORY_TURNS:]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        out.append(cls(content=m["content"]))
    return out


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.image(LOGO_URL, width=180)
    st.header("Settings")

    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("Groq API key", type="password",
                                help="Or set GROQ_API_KEY in .streamlit/secrets.toml")

    model_name = st.text_input("Groq model", value=DEFAULT_LLM)
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, 5)

    st.subheader("Knowledge sources")
    extra_pdfs = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("Add your Groq API key in the sidebar to start.")
    st.stop()

# ----------------------------------------------------------------------------
# Build knowledge base
# ----------------------------------------------------------------------------
uploaded = [(f.name, f.getvalue()) for f in (extra_pdfs or [])]

hasher = hashlib.sha256()
for p in sorted(PDF_DIR.glob("*.pdf")):
    hasher.update(p.name.encode()); hasher.update(str(p.stat().st_mtime).encode())
for name, data in uploaded:
    hasher.update(name.encode()); hasher.update(hashlib.sha256(data).digest())

with st.spinner("Loading website and documents… (first run downloads the embedding model)"):
    pages = scrape_website(tuple(WEBSITE_URLS))
    cache_key = hasher.hexdigest() + "".join(str(len(p["text"])) for p in pages)
    vector_store, n_chunks = build_vector_store(cache_key, pages, uploaded)

with st.sidebar:
    for p in pages:
        if p["error"]:
            st.warning(f"Could not fetch {p['url']}")
    st.caption(f"Indexed {n_chunks} chunks from {len(WEBSITE_URLS)} pages "
               f"and {len(list(PDF_DIR.glob('*.pdf'))) + len(uploaded)} PDFs.")

if vector_store is None:
    st.error("No content could be loaded. Check your internet connection or add PDFs.")
    st.stop()

retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
llm = ChatGroq(model=model_name, api_key=api_key, temperature=0)

# ----------------------------------------------------------------------------
# Chat UI
# ----------------------------------------------------------------------------
st.title("🎓 Full Stack Academy RAG Chatbot")
st.caption("Ask questions about Full Stack Academy or the uploaded documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

EXAMPLES = [
    "What courses does Full Stack Academy offer?",
    "Tell me about Mujeeb's experience at Persistent Systems.",
    "What was the previous question I asked you?",
]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"- {s}")

example_clicked = None
if not st.session_state.messages:
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            example_clicked = ex

question = st.chat_input("Ask me about Full Stack Academy…") or example_clicked

if question:
    history = to_lc_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            # 1. Rewrite follow-ups into a standalone search query
            search_query = question
            if history:
                search_query = (CONDENSE_PROMPT | llm | StrOutputParser()).invoke(
                    {"chat_history": history, "question": question}
                )

            # 2. Retrieve
            docs = retriever.invoke(search_query)

            # 3. Answer (streamed)
            chain = QA_PROMPT | llm | StrOutputParser()
            answer = st.write_stream(chain.stream({
                "context": format_docs(docs),
                "chat_history": history,
                "question": question,
            }))

            sources = sorted({
                f"{Path(str(d.metadata.get('source', 'unknown'))).name}"
                + (f" (page {d.metadata['page'] + 1})" if "page" in d.metadata else "")
                if not str(d.metadata.get("source", "")).startswith("http")
                else d.metadata["source"]
                for d in docs
            })
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
