import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence HF tokenizer warning

import json
import random
import re
from datetime import date
from flask import Flask, render_template, request
from pinecone import Pinecone, ServerlessSpec

# LlamaIndex / OpenAI for LlamaIndex
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.prompts import PromptTemplate

print("Loading OraculAI app v2.5 (sources-only QA + interpretive daily quote + hot refresh)")

# --- Flask app ---
app = Flask(__name__)

# --- Env vars ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")  # e.g., "us-east-1"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# --- LlamaIndex global settings ---
Settings.llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- STRICT sources-only QA prompt (used by /query) ---
STRICT_QA_PROMPT = PromptTemplate(
    """You answer questions using ONLY the provided context.
If the answer is not explicitly present in the context, reply exactly:
"I don’t have that in my sources yet."

Do not use outside knowledge. Do not speculate. Keep answers concise.

Context:
{context_str}

Question: {query_str}

Answer (from context only):"""
)

# --- INTERPRETIVE daily quote prompt (sources-grounded paraphrase) ---
INTERPRETIVE_QUOTE_PROMPT = PromptTemplate(
    """You will receive context chunks from the user's private sources.
Synthesize ONE concise, aphoristic sentence (your own words) that captures a key idea present in the context.

Rules:
- Ground strictly in the provided context; do not add outside knowledge.
- If the context is insufficient, reply exactly: "I don’t have that in my sources yet."
- Keep it under 220 characters.
- Do not include citations or metadata.

Context:
{context_str}

Instruction: Produce one paraphrased, reflective sentence grounded in the context.

Answer:"""
)

# --- Globals so /refresh can rebuild them ---
index = None
query_engine = None

# --- Pinecone helpers ---
def _existing_index_names(pc: Pinecone):
    idx = pc.list_indexes()
    if hasattr(idx, "names") and callable(idx.names):
        return set(idx.names())            # v3+
    if isinstance(idx, list):
        return set(idx)                    # list of names
    try:
        return {i["name"] for i in idx.get("indexes", [])}  # dict payload
    except Exception:
        return set()

def _build_index_and_engine():
    """(Re)build vector index from ./sources and return (index, strict_query_engine)."""
    try:
        if not PINECONE_API_KEY:
            raise RuntimeError("Missing PINECONE_API_KEY")
        if not PINECONE_ENVIRONMENT:
            raise RuntimeError("Missing PINECONE_ENVIRONMENT (region like 'us-east-1').")

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "oraculai-index"

        existing = _existing_index_names(pc)
        if index_name not in existing:
            print(f"[Pinecone] Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,  # all-MiniLM-L6-v2 embeddings dim
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
            )
            print("[Pinecone] Index create requested.")

        pinecone_index = pc.Index(index_name)

        # Load docs
        if not os.path.isdir("./sources"):
            print("[Indexing] ./sources folder not found. Create it and add content.")
            documents = []
        else:
            documents = SimpleDirectoryReader(input_dir="./sources").load_data()
            print(f"[Indexing] Loaded {len(documents)} documents from ./sources")

        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build/refresh index (idempotent; adds/updates embeddings)
        new_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        print(f"[Indexing] Index '{index_name}' ready.")

        # Strict engine for the /query route
        strict_engine = new_index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            text_qa_template=STRICT_QA_PROMPT,
        )
        return new_index, strict_engine

    except Exception as e:
        print(f"[ERROR] Pinecone setup/index build failed: {e}")
        return None, None

# Build once at startup
index, query_engine = _build_index_and_engine()

# --- Utilities ---
def _clean_quote(text: str) -> str:
    if not text:
        return ""
    q = str(text).strip()
    q = re.sub(r"\s+", " ", q)
    q = q.strip('"\u201c\u201d\u2018\u2019')
    if len(q) > 240:
        cut = q[:240]
        last_period = cut.rfind(".")
        q = cut[: last_period + 1] if last_period > 80 else cut + "…"
    return q

def _generate_image_url(prompt: str) -> str:
    """Force legacy DALL·E-3 path to avoid gpt-image-1 403 in unverified orgs."""
    if not OPENAI_API_KEY:
        print("[Image] OPENAI_API_KEY missing; cannot generate")
        return ""
    try:
        import openai as legacy_openai
        legacy_openai.api_key = OPENAI_API_KEY
        resp = legacy_openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        url = resp.data[0].url if getattr(resp, "data", None) else ""
        if url:
            print("[Image] Generated via legacy SDK (dall-e-3).")
        return url
    except Exception as e:
        print(f"[Image] Legacy SDK path failed: {e}")
        return ""

# --- Oracle QA (STRICT sources only, with citations) ---
def get_oracle_response(user_query: str):
    """
    Returns (answer_text, sources) where sources is a list of dicts:
    [{"file": <filename>, "score": <float>, "snippet": <text>}, ...]
    """
    global query_engine
    if query_engine is None:
        return (
            "Sorry, the oracle's memory is offline. "
            "Please check your API keys and the ingestion step."
        ), []

    try:
        result = query_engine.query(user_query)
        answer = str(result).strip()

        # Gather citations/snippets
        sources = []
        for sn in getattr(result, "source_nodes", [])[:5]:
            meta = getattr(sn, "metadata", {}) or {}
            file_name = meta.get("file_name") or meta.get("source") or meta.get("filename") or "Unknown source"
            snippet = sn.node.get_content() if hasattr(sn, "node") else ""
            sources.append({
                "file": file_name,
                "score": getattr(sn, "score", None),
                "snippet": snippet[:300].replace("\n", " ") + ("…" if len(snippet) > 300 else "")
            })

        return answer, sources
    except Exception as e:
        print(f"[Query] Error: {e}")
        return "I don’t have that in my sources yet.", []

# --- Daily content helpers ---
def _interpretive_quote_from_sources() -> str:
    """
    Paraphrased, sources-only daily quote (one sentence).
    Uses its own interpretive engine so /query stays strict.
    """
    global index
    if not index:
        return ""
    try:
        interpretive_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            text_qa_template=INTERPRETIVE_QUOTE_PROMPT,
        )
        result = interpretive_engine.query(
            "Create one reflective, aphoristic sentence grounded in the context."
        )
        quote = _clean_quote(str(result).strip())
        # filter strict fallback if model emitted it
        return quote if quote and quote != "I don’t have that in my sources yet." else ""
    except Exception as e:
        print(f"[Quote] Interpretive path failed: {e}")
        return ""

def _quote_from_file(quotes_file: str = "quotes.txt") -> str:
    try:
        with open(quotes_file, "r") as f:
            choices = [q.strip() for q in f.readlines() if q.strip()]
        return _clean_quote(random.choice(choices)) if choices else ""
    except Exception as e:
        print(f"[Quote] Error reading quotes.txt: {e}")
        return ""

def get_daily_content(force_refresh: bool = False):
    """
    Daily content pipeline:
      1) Interpretive, sources-grounded quote (paraphrase)
      2) Fallback: quotes.txt
      3) Final fallback: fixed string
      + image generation (DALL·E 3)
      + JSON cache (per day)
      + ?nocache=1 bypass
    """
    today = str(date.today())
    DAILY_CONTENT_FILE = "daily_content.json"

    # URL param override: /?nocache=1
    if request.args.get("nocache") in ("1", "true", "yes"):
        force_refresh = True

    def _cache_is_usable(payload: dict) -> bool:
        if not payload or payload.get("date") != today:
            return False
        q = (payload.get("quote") or "").strip()
        img = (payload.get("image_url") or "").strip()
        if q == "A random quote from the void." or not img:
            return False
        return True

    if not force_refresh and os.path.exists(DAILY_CONTENT_FILE):
        try:
            with open(DAILY_CONTENT_FILE, "r") as f:
                data = json.load(f)
            if _cache_is_usable(data):
                print("Using cached content for today.")
                return data.get("quote", ""), data.get("image_url", "")
            else:
                print("[Daily] Cache exists but is low-quality or stale; regenerating.")
        except Exception:
            print("[Daily] Cache read failed; regenerating.")

    print("[Daily] Generating new content (interpretive sources → file → fallback).")

    quote = _interpretive_quote_from_sources() or _quote_from_file() or "A random quote from the void."
    image_prompt = f"A high-quality spiritual image inspired by this quote: '{quote}'"
    image_url = _generate_image_url(image_prompt)

    new_content = {"date": today, "quote": quote, "image_url": image_url}
    try:
        with open(DAILY_CONTENT_FILE, "w") as f:
            json.dump(new_content, f)
        print("[Daily] Cached content for today.")
    except Exception as e:
        print(f"[Daily] Error caching daily content: {e}")

    return quote, image_url

# --- Routes ---
@app.route("/")
def home():
    daily_quote, daily_image_url = get_daily_content(force_refresh=False)
    return render_template("index.html", quote=daily_quote, image_url=daily_image_url)

@app.route("/regen")
def regen():
    daily_quote, daily_image_url = get_daily_content(force_refresh=True)
    return render_template("index.html", quote=daily_quote, image_url=daily_image_url)

@app.route("/refresh")
def refresh_index():
    """Hot-reload ./sources into Pinecone without restarting the server."""
    global index, query_engine
    index, query_engine = _build_index_and_engine()
    if query_engine is None:
        return "Index refresh failed. Check server logs.", 500
    return "Index refreshed with new sources."

@app.route("/query", methods=["POST"])
def query_oracle():
    user_query = request.form.get("user_query", "").strip()
    if not user_query:
        return render_template("response.html", query=user_query, response="Please enter a question.", sources=[])
    answer, sources = get_oracle_response(user_query)
    return render_template("response.html", query=user_query, response=answer, sources=sources)

# --- Main ---
if __name__ == "__main__":
    app.run(debug=True)