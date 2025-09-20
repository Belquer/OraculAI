"""Flask app powering OraculAI."""

from __future__ import annotations

import json
import os
import random
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence HF tokenizer warning

from flask import Flask, render_template, request

try:  # ensure runtime surfaces actionable guidance when extras are missing
    from pinecone import Pinecone, ServerlessSpec
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing optional dependency 'pinecone'. Install it via 'pip install pinecone'."
    ) from exc

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.prompts import PromptTemplate

# Heavy ML / vector-store imports are performed lazily inside the index build
# function to keep the webserver start fast. See _build_index_and_engine().

print("Loading OraculAI app (sources-only QA + interpretive daily quote)")

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")  # e.g. "us-east-1"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Settings.llm and Settings.embed_model are configured lazily during index
# build so the webserver avoids heavy ML imports at startup.

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

index: Optional[VectorStoreIndex] = None
query_engine = None

# Chunking tuning for ingestion (chars, not strict tokens). We prefer larger
# chunks to keep runtime vector counts low and retrieval fast. These constants
# are used by the ingestion path; they do not force heavy imports at server
# startup.
CHUNK_MAX_CHARS = 4000  # ~ 500-800 tokens depending on content
CHUNK_OVERLAP = 200


def _existing_index_names(pc: Pinecone) -> set[str]:
    idx = pc.list_indexes()
    if hasattr(idx, "names") and callable(idx.names):
        return set(idx.names())
    if isinstance(idx, list):
        return set(idx)
    try:
        return {item["name"] for item in idx.get("indexes", [])}
    except Exception:
        return set()


def _build_index_and_engine() -> Tuple[Optional[VectorStoreIndex], Any]:
    """(Re)build vector index from ./sources and return (index, strict_query_engine)."""
    try:
        # Lazy imports: bring in heavy or optional dependencies only when building
        # the index so the webserver startup remains fast.
        try:
            from llama_index.embeddings.huggingface import (
                HuggingFaceEmbedding,
            )
            from llama_index.llms.openai import OpenAI as LlamaOpenAI
            from llama_index.vector_stores.pinecone import PineconeVectorStore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Missing llama-index extras for embedding/vector-store. Install 'llama-index-embeddings-huggingface' and 'llama-index-vector-stores-pinecone'"
            ) from exc

        # Configure LLM and embedding model for index build
        Settings.llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if not PINECONE_API_KEY:
            raise RuntimeError("Missing PINECONE_API_KEY")
        if not PINECONE_ENVIRONMENT:
            raise RuntimeError(
                "Missing PINECONE_ENVIRONMENT (region like 'us-east-1')."
            )

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "oraculai-index"

        if index_name not in _existing_index_names(pc):
            print(f"[Pinecone] Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
            )
            print("[Pinecone] Index create requested.")

        pinecone_index = pc.Index(index_name)

        if not os.path.isdir("./sources"):
            print("[Indexing] ./sources folder not found. Create it and add content.")
            documents = []
        else:
            documents = SimpleDirectoryReader(input_dir="./sources").load_data()
            print(f"[Indexing] Loaded {len(documents)} documents from ./sources")

        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        built_index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        print(f"[Indexing] Index '{index_name}' ready.")

        strict_engine = built_index.as_query_engine(
            similarity_top_k=4,
            response_mode="compact",
            text_qa_template=STRICT_QA_PROMPT,
        )
        return built_index, strict_engine
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[ERROR] Pinecone setup/index build failed: {exc}")
        return None, None


# NOTE: defer building the Pinecone/llama-index vector index until an explicit
# refresh is requested. This prevents long import-time delays when heavy
# dependencies (HF models, pinecone) are missing or slow to initialize.

# --- DEV fallback: provide a simple query engine when real index is unavailable
class _DevQueryEngine:
    def query(self, q: str):
        class R:
            def __str__(self):
                return "(DEV) I can't access the vector index right now. This is a placeholder response."

            source_nodes = []

        return R()

if query_engine is None:
    print("[DEV] Query engine unavailable; using dev fallback engine for initial testing.")
    query_engine = _DevQueryEngine()


def _clean_quote(text: str) -> str:
    if not text:
        return ""
    quote = str(text).strip()
    quote = re.sub(r"\s+", " ", quote)
    quote = quote.strip('"\u201c\u201d\u2018\u2019')
    if len(quote) > 240:
        truncated = quote[:240]
        last_period = truncated.rfind(".")
        quote = (
            truncated[: last_period + 1]
            if last_period > 80
            else truncated + "…"
        )
    return quote


def get_oracle_response(user_query: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (answer_text, sources payload)."""
    global query_engine
    if query_engine is None:
        return (
            "Sorry, the oracle's memory is offline. "
            "Please check your API keys and the ingestion step."
        ), []
    try:
        result = query_engine.query(user_query)
        answer = str(result).strip()

        sources: List[Dict[str, Any]] = []
        for source_node in getattr(result, "source_nodes", [])[:5]:
            metadata = getattr(source_node, "metadata", {}) or {}
            file_name = (
                metadata.get("file_name")
                or metadata.get("source")
                or metadata.get("filename")
                or "Unknown source"
            )
            snippet = (
                source_node.node.get_content() if hasattr(source_node, "node") else ""
            )
            sources.append(
                {
                    "file": file_name,
                    "score": getattr(source_node, "score", None),
                    "snippet": snippet[:300].replace("\n", " ")
                    + ("…" if len(snippet) > 300 else ""),
                }
            )
        return answer, sources
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[Query] Error: {exc}")
        return "I don’t have that in my sources yet.", []


def _interpretive_quote_from_sources() -> str:
    """Build interpretive quote via separate query engine so strict QA stays safe."""
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
        if quote and quote != "I don’t have that in my sources yet.":
            return quote
        return ""
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[Quote] Interpretive path failed: {exc}")
        return ""


def _quote_from_file(quotes_file: str = "quotes.txt") -> str:
    try:
        with open(quotes_file, "r", encoding="utf-8") as file:
            choices = [line.strip() for line in file if line.strip()]
        return _clean_quote(random.choice(choices)) if choices else ""
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[Quote] Error reading {quotes_file}: {exc}")
        return ""


def get_daily_content(force_refresh: bool = False) -> Tuple[str, str]:
    """Load or regenerate daily quote with caching. Image generation is disabled."""
    today = str(date.today())
    cache_path = "daily_content.json"

    if request.args.get("nocache") in ("1", "true", "yes"):
        force_refresh = True

    def _cache_is_usable(payload: Dict[str, Any]) -> bool:
        if not payload or payload.get("date") != today:
            return False
        quote = (payload.get("quote") or "").strip()
        return bool(quote and quote != "A random quote from the void.")

    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
            if _cache_is_usable(payload):
                print("Using cached content for today.")
                return payload.get("quote", ""), ""
            print("[Daily] Cache exists but is low-quality or stale; regenerating.")
        except Exception:
            print("[Daily] Cache read failed; regenerating.")

    print("[Daily] Generating new content (interpretive sources → file → fallback).")

    quote = (
        _interpretive_quote_from_sources()
        or _quote_from_file("sources/quotes.txt")
        or _quote_from_file("quotes.txt")
        or "A random quote from the void."
    )

    payload = {"date": today, "quote": quote}
    try:
        with open(cache_path, "w", encoding="utf-8") as cache_file:
            json.dump(payload, cache_file)
        print("[Daily] Cached content for today.")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[Daily] Error caching daily content: {exc}")

    return quote, ""


@app.route("/")
def home():
    daily_quote, daily_image_url = get_daily_content(force_refresh=False)
    return render_template(
        "index.html", quote=daily_quote, image_url=daily_image_url, env=os.environ
    )


@app.route("/regen")
def regen():
    daily_quote, daily_image_url = get_daily_content(force_refresh=True)
    return render_template(
        "index.html", quote=daily_quote, image_url=daily_image_url, env=os.environ
    )


@app.route("/ask")
def ask():
    return render_template("ask.html")


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
        return render_template(
            "response.html",
            query=user_query,
            response="Please enter a question.",
            sources=[],
        )
    answer, sources = get_oracle_response(user_query)
    return render_template(
        "response.html", query=user_query, response=answer, sources=sources
    )


if __name__ == "__main__":
    app.run(debug=True)

# create .venv if missing or to start fresh
# python3 -m venv .venv

# activate and install
# source .venv/bin/activate
# python -m pip install --upgrade pip
# pip install -r requirements.txt

# activate .venv first
# To refresh requirements, run this in a shell (not in Python):
# pip freeze | sed '/^-e /d' > requirements.txt
