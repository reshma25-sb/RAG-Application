# app.py
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# Embeddings & models
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Chroma
import chromadb
from chromadb.config import Settings

# Document loaders
from pypdf import PdfReader

# Text splitting (simple)
import math
import itertools

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path("data")
PERSIST_DIR = Path("chroma_db")
CHROMA_COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers
GENERATION_MODEL = "google/flan-t5-small"   # small T5-style model for generation

# Chunking params
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# -----------------------
# Utils: load files
# -----------------------
def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    text = []
    try:
        reader = PdfReader(str(path))
        for page in reader.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                # fallback continue
                continue
    except Exception as e:
        st.warning(f"Failed reading PDF {path}: {e}")
    return "\n".join(text)

def load_documents_from_folder(folder: Path) -> List[Dict[str, Any]]:
    """Return list of dicts { 'id','text','source' } from .txt and .pdf in folder"""
    docs = []
    folder.mkdir(parents=True, exist_ok=True)
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in [".txt", ".pdf"]:
            if p.suffix.lower() == ".txt":
                content = read_txt(p)
            else:
                content = read_pdf(p)
            if content and content.strip():
                docs.append({"id": str(p.name), "text": content, "source": str(p.name)})
    return docs

# -----------------------
# Utils: text split
# -----------------------
def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end >= len(text):
            break
    return chunks

def docs_to_chunks(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    idx = 0
    for d in docs:
        text = d["text"]
        paragraphs = text.split("\n\n") if "\n\n" in text else [text]
        # For each paragraph produce chunks
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            for c in split_text(para):
                chunks.append({
                    "id": f"{d['id']}_chunk_{idx}",
                    "text": c,
                    "source": d["source"]
                })
                idx += 1
    return chunks

# -----------------------
# Initialize embeddings and generator (lazily)
# -----------------------
@st.cache_resource
def get_embedding_model():
    # load sentence-transformers model (local)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def get_generation_pipeline():
    # use transformers seq2seq pipeline (small model)
    # loads tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL)
    gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU
    return gen

# -----------------------
# Chroma client helpers
# -----------------------
def get_chroma_client(persist_directory: Path = PERSIST_DIR):
    # Use duckdb+parquet persistent mode
    # chromadb v0.5+ supports PersistentClient; older versions use chromadb.Client with Settings
    try:
        client = chromadb.PersistentClient(path=str(persist_directory))
    except Exception:
        # fallback to legacy Client
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(persist_directory)))
    return client

def create_or_get_collection(client, name: str = CHROMA_COLLECTION_NAME):
    # create or get; different clients have different methods
    if hasattr(client, "get_collection"):
        try:
            return client.get_collection(name)
        except Exception:
            return client.create_collection(name)
    else:
        # PersistentClient has get_or_create_collection
        try:
            return client.get_or_create_collection(name=name)
        except Exception:
            return client.create_collection(name)

# -----------------------
# Indexing
# -----------------------
def index_documents(force_reindex: bool = False) -> Dict[str, Any]:
    """Load files from data/, chunk, embed, and upsert into Chroma. Returns stats."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_documents_from_folder(DATA_DIR)
    if not docs:
        return {"status": "no_documents", "count": 0}

    chunks = docs_to_chunks(docs)
    if not chunks:
        return {"status": "no_chunks", "count": 0}

    # embeddings
    embed_model = get_embedding_model()
    texts = [c["text"] for c in chunks]

    # compute embeddings in batches to avoid memory spike
    batch_size = 128
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = embed_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
        embeddings.extend(emb)

    # connect to chroma and upsert
    client = get_chroma_client(PERSIST_DIR)
    # prefer get_or_create
    try:
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        # older API
        collection = client.create_collection(CHROMA_COLLECTION_NAME)

    ids = [c["id"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]
    documents = [c["text"] for c in chunks]

    # Remove existing collection contents if reindex requested
    if force_reindex:
        try:
            collection.delete(ids=collection.get()["ids"])
        except Exception:
            # best effort
            pass

    # upsert (add)
    # APIs vary slightly; we'll try known methods
    try:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
    except TypeError:
        # some versions expect 'documents' only as second param etc.
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    # persist if client supports it
    try:
        client.persist()
    except Exception:
        pass

    return {"status": "indexed", "chunks": len(chunks), "documents": len(docs)}

# -----------------------
# Retrieval + generation
# -----------------------
def retrieve_top_k(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """Return top-k retrieved docs from Chroma for query."""
    embed_model = get_embedding_model()
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0].tolist()

    client = get_chroma_client(PERSIST_DIR)
    try:
        collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)

    # Many chroma versions accept query with embeddings
    try:
        result = collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=['metadatas', 'documents', 'distances']
        )
        # result structure may vary; handle common case
        docs = []
        if "documents" in result:
            for doc_text, meta, dist in zip(result["documents"][0], result["metadatas"][0], result.get("distances",[[]])[0]):
                docs.append({"text": doc_text, "metadata": meta, "distance": dist})
        else:
            # older response shape
            docs = []
        return docs
    except Exception as e:
        # fallback: some APIs use query with 'n' or different names
        try:
            result = collection.query(
                embeddings=[q_emb],
                n_results=k
            )
            docs = []
            for doc_text, meta in zip(result["documents"][0], result["metadatas"][0]):
                docs.append({"text": doc_text, "metadata": meta, "distance": None})
            return docs
        except Exception as e2:
            st.error(f"Chroma query error: {e} / {e2}")
            return []

@st.cache_resource
def get_generator():
    return get_generation_pipeline()

def generate_answer(question: str, retrieved: List[Dict[str, Any]]) -> str:
    """Create prompt using retrieved contexts and call generator pipeline."""
    generator = get_generator()

    # build context: top-k passages concatenated with source tags
    pieces = []
    for r in retrieved:
        md = r.get("metadata") or {}
        source = md.get("source", "unknown")
        pieces.append(f"[Source: {source}]\n{r['text'].strip()}")
    context = "\n\n---\n\n".join(pieces) if pieces else ""

    prompt = (
        "Use the following context to answer the question. If the answer is not present, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    # Generate (may return multiple candidates). Keep max_length reasonable.
    try:
        out = generator(prompt, max_length=256, do_sample=False)
        if isinstance(out, list) and len(out) > 0:
            return out[0]["generated_text"]
        else:
            return str(out)
    except Exception as e:
        # fallback: return concatenation of top docs if generator fails
        return "Failed to generate answer: " + str(e) + "\n\nContext used:\n" + context[:1000]

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="RAG Q & A (LangChain + Chroma)", layout="wide")
st.title("📚 RAG Document Question & Answer — LangChain-style ")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Indexing / Documents")
    st.write("Drop .txt or .pdf files into the `data/` folder (server-side).")
    if st.button("🔄 Index documents"):
        with st.spinner("Indexing documents (this may take a minute)..."):
            res = index_documents(force_reindex=False)
            st.success(f"Index result: {res}")

    uploaded = st.file_uploader("Or upload a file (txt/pdf)", type=["txt", "pdf"])
    if uploaded is not None:
        save_path = DATA_DIR / uploaded.name
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved {uploaded.name}")
        if st.button("Index after upload"):
            with st.spinner("Indexing..."):
                res = index_documents(force_reindex=False)
                st.success(f"Index result: {res}")

with col2:
    st.header("Ask a question")
    question = st.text_input("Enter your question about the documents:")
    top_k = st.slider("Top k retrieved passages", 1, 8, 4)
    if st.button("Ask"):
        if not question or question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant documents..."):
                retrieved = retrieve_top_k(question, k=top_k)
            if not retrieved:
                st.info("No context found. Index some documents first.")
            else:
                st.write("#### Retrieved passages (top results):")
                for i, r in enumerate(retrieved[:top_k]):
                    src = r.get("metadata", {}).get("source", "unknown")
                    st.markdown(f"**{i+1}. Source:** {src} — *distance*={r.get('distance')}")
                    st.write(r["text"][:1000] + ("..." if len(r["text"]) > 1000 else ""))
                with st.spinner("Generating answer..."):
                    answer = generate_answer(question, retrieved)
                st.markdown("### 🧠 Answer")
                st.write(answer)

st.markdown("---")
st.caption("This app computes local embeddings (sentence-transformers) and stores them in Chroma. Generation uses a local transformers model (Flan-T5 small).")
