import os
import re
from datetime import datetime, timedelta
from flask import Flask, render_template
from pinecone import Pinecone, ServerlessSpec

# LlamaIndex
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.prompts import PromptTemplate

# ── Version & app ─────────────────────────────────────────────────────────────
VERSION = "2.10.1"  # semantic version
APP_PATH = os.path.abspath(__file__)

app = Flask(__name__)

print(f"Loading OraculAI app v{VERSION} (sources-only QA + interpretive daily quote + hot refresh)")
print(f"[Startup] Running from: {APP_PATH}")

# ── Env & LlamaIndex settings ────────────────────────────────────────────────
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")  # e.g., "us-east-1"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("ORACULAI_INDEX_NAME", "oraculai-index-384")  # NEW: 384-dim index

# Configure LlamaIndex globals
Settings.llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")  # 384-dim

# ── Prompts ──────────────────────────────────────────────────────────────────
INTERPRETIVE_QUOTE_PROMPT = PromptTemplate(
    """You will receive context chunks from the user's private sources.
Synthesize ONE concise, aphoristic sentence (your own words) that captures a key idea present in the context.

Rules:
- Ground strictly in the provided context; do not add outside knowledge.
- If the context is insufficient, reply exactly: "I don’t have that in my sources yet."
- Keep it under 220 characters.

Context:
{context_str}

Instruction: Produce one paraphrased, reflective sentence grounded in the context.

Answer:"""
)

# ── Globals: index + cache ───────────────────────────────────────────────────
_index = None
_query_engine_interpretive = None
_last_quote = None
_last_quote_at = None
QUOTE_TTL = int(os.environ.get("ORACULAI_QUOTE_TTL_SECONDS", "600"))  # 10 min default

# ── Helpers ──────────────────────────────────────────────────────────────────
def _existing_index_names(pc: Pinecone):
    """Return a set of existing index names across SDK variants."""
    idx = pc.list_indexes()
    if hasattr(idx, "names") and callable(idx.names):
        return set(idx.names())
    if isinstance(idx, list):
        return set(idx)
    try:
        return {i["name"] for i in idx.get("indexes", [])}
    except Exception:
        return set()

def _clean_quote(text: str) -> str:
    if not text:
        return ""
    q = re.sub(r"\s+", " ", str(text).strip())
    q = q.strip('"\u201c\u201d\u2018\u2019')
    return q[:260].rstrip() + ("…" if len(q) > 260 else "")

def _build_index():
    """(Re)build the Pinecone-backed index from ./sources using 384-dim embeddings."""
    global _index, _query_engine_interpretive

    if not PINECONE_API_KEY:
        print("[ERROR] Missing PINECONE_API_KEY")
        _index = None
        _query_engine_interpretive = None
        return

    if not PINECONE_ENVIRONMENT:
        print("[ERROR] Missing PINECONE_ENVIRONMENT (e.g., 'us-east-1')")
        _index = None
        _query_engine_interpretive = None
        return

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = INDEX_NAME  # 384-dim index name

        if index_name not in _existing_index_names(pc):
            print(f"[Pinecone] Creating index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,  # all-MiniLM-L6-v2 (keep 384 here)
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
            )

        pinecone_index = pc.Index(index_name)

        # Load docs from ./sources
        if not os.path.isdir("./sources"):
            print("[Indexing] ./sources folder not found.")
            documents = []
        else:
            documents = SimpleDirectoryReader(input_dir="./sources").load_data()
            print(f"[Indexing] Loaded {len(documents)} documents from ./sources")

        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        _index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        _query_engine_interpretive = _index.as_query_engine(
            similarity_top_k=6,
            response_mode="compact",
            text_qa_template=INTERPRETIVE_QUOTE_PROMPT,
        )
        print(f"[Indexing] Index '{index_name}' ready.")
    except Exception as e:
        print(f"[ERROR] Index build failed: {e}")
        _index = None
        _query_engine_interpretive = None

def _get_sources_quote(force=False) -> str:
    """Return an interpretive, sources-only quote with simple TTL caching."""
    global _last_quote, _last_quote_at, _query_engine_interpretive

    if not _query_engine_interpretive:
        return "I don’t have that in my sources yet."

    now = datetime.utcnow()
    if (
        not force
        and _last_quote is not None
        and _last_quote_at is not None
        and (now - _last_quote_at) < timedelta(seconds=QUOTE_TTL)
    ):
        return _last_quote

    try:
        res = _query_engine_interpretive.query(
            "Create one reflective, aphoristic sentence grounded in the context."
        )
        quote = _clean_quote(str(res).strip())
        if not quote or quote == "I don’t have that in my sources yet.":
            quote = "I don’t have that in my sources yet."
        _last_quote, _last_quote_at = quote, now
        return quote
    except Exception as e:
        print(f"[Quote] Error: {e}")
        return "I don’t have that in my sources yet."

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # Build index once lazily on first request
    global _index
    if _index is None:
        _build_index()

    quote = _get_sources_quote(force=False)
    if quote == "I don’t have that in my sources yet." and not (PINECONE_API_KEY and PINECONE_ENVIRONMENT and OPENAI_API_KEY):
        quote = "Add PDFs to ./sources and set your API keys, then visit /refresh."
    return render_template("index.html", quote=quote)

@app.route("/refresh")
def refresh():
    """Hot-reload ./sources into Pinecone and clear the quote cache."""
    global _last_quote, _last_quote_at
    _build_index()
    _last_quote = None
    _last_quote_at = None
    ok = (_index is not None) and (_query_engine_interpretive is not None)
    return ("Index refreshed." if ok else "Index refresh failed. Check logs."), (200 if ok else 500)

@app.route("/version")
def version():
    return f"OraculAI v{VERSION} · {APP_PATH}"

# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)