"""Flask app powering OraculAI."""

from __future__ import annotations

import json
import os
import random
import re
import pathlib
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence HF tokenizer warning

from flask import Flask, render_template, request

# Soft import pinecone so the webserver can start even if it's not installed.
# We'll re-import inside the index build path and handle errors there.
try:  # pragma: no cover - import guard
    from pinecone import Pinecone, ServerlessSpec  # type: ignore
except Exception:
    Pinecone = None  # type: ignore
    ServerlessSpec = None  # type: ignore

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

# Poetic answer prompt: used when the strict QA engine finds no direct answer.
# In that case we synthesise a short, poetic reflection grounded ONLY in the
# retrieved context; do not invent facts or add outside knowledge.
POETIC_ANSWER_PROMPT = PromptTemplate(
    """You will receive context chunks from the user's private sources and a
question. Using ONLY the provided context, write a concise (1-3 sentences)
poetic answer to the question. Ground every turn of phrase in the context;
do not invent facts or hallucinate. If the context does not directly answer the
question, create a reflective, paraphrased response that explores how the
sources relate to the question without asserting new facts.

Context:
{context_str}

Question: {query_str}

Answer (1-3 poetic sentences, grounded in context):"""
)

index: Optional[VectorStoreIndex] = None
query_engine = None
engine_backend = "dev"  # one of: dev | pinecone | local (future)

# Chunking tuning for ingestion (chars, not strict tokens). We prefer larger
# chunks to keep runtime vector counts low and retrieval fast. These constants
# are used by the ingestion path; they do not force heavy imports at server
# startup.
CHUNK_MAX_CHARS = 4000  # ~ 500-800 tokens depending on content
CHUNK_OVERLAP = 200


from typing import Any as _Any


def _existing_index_names(pc: _Any) -> set[str]:
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
    import sys
    print("[BUILD] Starting _build_index_and_engine function", flush=True)
    sys.stdout.flush()
    
    try:
        print("[BUILD] About to perform lazy imports", flush=True)
        sys.stdout.flush()
        # Lazy imports: bring in heavy or optional dependencies only when building
        # the index so the webserver startup remains fast.
        try:
                print("[BUILD] Importing llama-index components", flush=True)
                sys.stdout.flush()
                from llama_index.embeddings.huggingface import (
                    HuggingFaceEmbedding,
                )
                from llama_index.llms.openai import OpenAI as LlamaOpenAI
                # Prefer OpenAI embedding adapter when available so the server
                # uses the same embedding model as the ingestion pipeline
                try:
                    from llama_index.embeddings.openai import OpenAIEmbedding
                    print("[BUILD] OpenAI embedding adapter imported successfully", flush=True)
                    sys.stdout.flush()
                except Exception:
                    print("[BUILD] OpenAI embedding adapter not available", flush=True)
                    sys.stdout.flush()
                    OpenAIEmbedding = None
                from llama_index.vector_stores.pinecone import PineconeVectorStore
                print("[BUILD] All llama-index imports completed", flush=True)
                sys.stdout.flush()
        except ImportError as exc:  # pragma: no cover - import guard
            print(f"[BUILD] Import error: {exc}", flush=True)
            sys.stdout.flush()
            raise ImportError(
                "Missing llama-index extras for embedding/vector-store. Install 'llama-index-embeddings-huggingface' and 'llama-index-vector-stores-pinecone'"
            ) from exc

        # (Re)import pinecone here to fail gracefully at build time rather than import time
        global Pinecone, ServerlessSpec
        if Pinecone is None:
            print("[BUILD] Importing pinecone client", flush=True)
            sys.stdout.flush()
            try:
                from pinecone import Pinecone as _Pinecone, ServerlessSpec as _ServerlessSpec  # type: ignore
                Pinecone, ServerlessSpec = _Pinecone, _ServerlessSpec
                print("[BUILD] Pinecone client imported successfully", flush=True)
                sys.stdout.flush()
            except Exception as e:
                print(f"[BUILD] Pinecone import failed: {e}", flush=True)
                sys.stdout.flush()
                raise RuntimeError(
                    "Missing optional dependency 'pinecone'. Install it via 'pip install pinecone'."
                ) from e

        # Configure LLM and embedding model for index build
        print(f"[BUILD] Configuring LLM with OpenAI key: {OPENAI_API_KEY[:10] if OPENAI_API_KEY else 'None'}...", flush=True)
        sys.stdout.flush()
        Settings.llm = LlamaOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
        print("[BUILD] LLM configured successfully", flush=True)
        sys.stdout.flush()
        
        # Prefer the OpenAI embedding adapter (text-embedding-3-small -> 1536 dims)
        print(f"[BUILD] Configuring embeddings. OpenAI available: {OpenAIEmbedding is not None}, API key present: {OPENAI_API_KEY is not None}", flush=True)
        sys.stdout.flush()
        if OpenAIEmbedding is not None and OPENAI_API_KEY:
            try:
                print("[BUILD] Attempting to initialize OpenAI embeddings", flush=True)
                sys.stdout.flush()
                Settings.embed_model = OpenAIEmbedding(
                    model_name=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                    api_key=OPENAI_API_KEY,
                )
                print("[BUILD] OpenAI embeddings configured successfully", flush=True)
                sys.stdout.flush()
            except Exception as e:
                print(f"[BUILD] OpenAI embedding initialization failed: {e}", flush=True)
                sys.stdout.flush()
                # if the OpenAI adapter is present but initialization fails,
                # fall back to the HF embedding to keep the server usable
                if os.environ.get("ORACULAI_ALLOW_HF_EMBED") == "1":
                    print("[BUILD] Falling back to HuggingFace model (ORACULAI_ALLOW_HF_EMBED=1)", flush=True)
                    sys.stdout.flush()
                    Settings.embed_model = HuggingFaceEmbedding(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                else:
                    print("[BUILD] HF fallback disabled. No embedding model configured.", flush=True)
                    sys.stdout.flush()
                    Settings.embed_model = None # Explicitly disable embeddings
        else:
            if os.environ.get("ORACULAI_ALLOW_HF_EMBED") == "1":
                print("[Embed] Using HuggingFace model (ORACULAI_ALLOW_HF_EMBED=1)")
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            else:
                print("[Embed] OpenAI embedding provider not found and HF fallback is disabled. No embedding model is configured.")
                Settings.embed_model = None # Explicitly disable embeddings

        if Settings.embed_model is None:
            raise RuntimeError("Embedding model could not be configured. Check OpenAI keys and/or ORACULAI_ALLOW_HF_EMBED setting.")

        if not PINECONE_API_KEY:
            raise RuntimeError("Missing PINECONE_API_KEY")
        # PINECONE_ENVIRONMENT may be a region like 'us-east-1' or a compound
        # value used by some Pinecone SDKs. Prefer passing it through to the
        # client if present but don't require it for read-only index use.
        if not PINECONE_ENVIRONMENT:
            print("[Pinecone] Warning: PINECONE_ENVIRONMENT not set; attempting client init without environment")

        # initialize Pinecone client with environment when supported
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        except TypeError:
            pc = Pinecone(api_key=PINECONE_API_KEY)

        # Respect environment variable for index name, fallback to 'oraculai'
        index_name = os.environ.get("PINECONE_INDEX_NAME") or os.environ.get("PINECONE_INDEX") or "oraculai"

        # If the index doesn't exist, try to create it with 1536-dim cosine (matches OpenAI text-embedding-3-small)
        if index_name not in _existing_index_names(pc):
            print(f"[Pinecone] Creating new index: {index_name}")
            dim = int(os.environ.get("EMBEDDING_DIM", "1536"))
            metric = os.environ.get("PINECONE_METRIC", "cosine")
            try:
                pc.create_index(name=index_name, dimension=dim, metric=metric)
                print("[Pinecone] Index create requested.")
            except Exception:
                # Fallback to ServerlessSpec path if plain create fails and ServerlessSpec is available
                try:
                    spec = ServerlessSpec(cloud=os.environ.get("PINECONE_CLOUD", "aws"), region=PINECONE_ENVIRONMENT)
                    pc.create_index(name=index_name, dimension=dim, metric=metric, spec=spec)
                    print("[Pinecone] Index create requested via ServerlessSpec.")
                except Exception as e:
                    raise RuntimeError(f"Failed to create Pinecone index {index_name}: {e}") from e

        pinecone_index = pc.Index(index_name)

        # Build a PineconeVectorStore compatible with multiple llama-index versions
        vector_store = None
        last_err = None
        try:
            # Newer signature (common): pass the Index instance
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)  # type: ignore[arg-type]
        except Exception as e1:
            last_err = e1
            try:
                # Alternate signature: via index_name and client
                vector_store = PineconeVectorStore(index_name=index_name, pinecone_client=pc)  # type: ignore[call-arg]
            except Exception as e2:
                last_err = e2
        if vector_store is None:
            raise RuntimeError(f"Failed to create PineconeVectorStore: {last_err}")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        skip_build = os.environ.get("ORACULAI_SKIP_BUILD") == "1"
        if skip_build:
            print(
                "[Indexing] ORACULAI_SKIP_BUILD=1 — loading existing Pinecone index without re-ingestion."
            )
            try:
                built_index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store, storage_context=storage_context
                )
                print(f"[Indexing] Attached to existing index '{index_name}'.")
            except Exception as exc:
                raise RuntimeError(
                    "Failed to attach to existing Pinecone index."
                ) from exc
        else:
            if not os.path.isdir("./sources"):
                print("[Indexing] ./sources folder not found. Create it and add content.")
                documents = []
            else:
                documents = SimpleDirectoryReader(input_dir="./sources").load_data()
                print(f"[Indexing] Loaded {len(documents)} documents from ./sources")

            built_index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            print(f"[Indexing] Index '{index_name}' ready.")

        strict_engine = built_index.as_query_engine(
            similarity_top_k=4,
            response_mode="compact",
            text_qa_template=STRICT_QA_PROMPT,
        )
        try:
            globals()["engine_backend"] = "pinecone"
        except Exception:
            pass
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


def _is_dev_engine(obj: Any) -> bool:
    try:
        return isinstance(obj, _DevQueryEngine)
    except Exception:
        return False


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
    # If the query engine hasn't been built in this process, attempt to
    # (re)build it now. This ensures the running worker has access to the
    # Pinecone-backed engine even when Flask's debug reloader has restarted
    # the process.
    if query_engine is None or _is_dev_engine(query_engine):
        try:
            built_index, built_engine = _build_index_and_engine()
            # only overwrite global state if build succeeded
            if built_engine:
                globals()["index"] = built_index
                globals()["query_engine"] = built_engine
                query_engine = built_engine
        except Exception as e:
            print(f"[Query] On-demand index build failed: {e}")

    if query_engine is None:
        # As a last resort, synthesize a poetic paraphrase directly from
        # the local files in ./sources. We'll pick full paragraphs (not
        # TOC/metadata), normalize them, and produce a short 1-2 sentence
        # poetic paraphrase.
        candidates: list[str] = []
        try:
            src_dir = pathlib.Path("./sources")
            if src_dir.exists():
                for p in sorted(src_dir.glob("*.txt")):
                    try:
                        text = p.read_text(encoding="utf-8")
                    except Exception:
                        continue
                    # split into paragraphs by blank lines
                    paras = [pp.strip() for pp in re.split(r"\n\s*\n", text) if pp.strip()]
                    for para in paras:
                        # skip short or boilerplate paragraphs
                        t = re.sub(r"\s+", " ", para)
                        if len(t) < 120:
                            continue
                        low = t.lower()
                        if any(k in low for k in ("table of contents", "copyright", "acknowledg", "chapter", "introduction", "contents")):
                            continue
                        if re.search(r"([^\w\s])\1{4,}", t):
                            continue
                        # prefer paragraphs with sentence punctuation
                        if not re.search(r"[\.\!\?]", t):
                            continue
                        candidates.append(t)
                        if len(candidates) >= 3:
                            break
                    if len(candidates) >= 3:
                        break
        except Exception:
            candidates = []

        if candidates:
            # take first candidate paragraph, extract first 1-2 sentences
            first = candidates[0]
            sents = re.split(r"(?<=[.!?])\s+", first)
            chosen = " ".join(sents[:2]).strip()
            chosen = re.sub(r"\s+", " ", chosen)
            if len(chosen) > 220:
                chosen = chosen[:220]
                last_space = chosen.rfind(" ")
                if last_space > 100:
                    chosen = chosen[:last_space]
                chosen = chosen.rstrip(".,;:!?") + "…"
            poetic = f"A poetic reading: {chosen}"
            return poetic, []

        # No good paragraph candidates found locally — still return a
        # cautious, source-grounded abstraction (avoid error messages).
        # We'll return a short, general reading inviting reflection.
        return (
            "Based on the texts available, a careful reading leans toward a"
            " reflective, open-ended stance on that question.",
            [],
        )

    try:
        # First try the strict QA engine
        result = query_engine.query(user_query)
        answer = str(result).strip()

        # Collect source snippets
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

        # If the strict engine gives the explicit 'not-in-sources' phrase or
        # an empty/unsatisfactory answer, synthesize a grounded poetic reply.
        forbidden = "I don’t have that in my sources yet."
        if not answer or answer.strip() == forbidden:
            # Build a secondary interpretive engine from the index if available
            global index
            if index:
                try:
                    poetic_engine = index.as_query_engine(
                        similarity_top_k=6,
                        response_mode="compact",
                        text_qa_template=POETIC_ANSWER_PROMPT,
                    )
                    poetic_result = poetic_engine.query(user_query)
                    try:
                        poetic_answer = str(poetic_result).strip()
                    except Exception:
                        # best-effort string conversion
                        poetic_answer = ""
                    if poetic_answer:
                        return poetic_answer, sources
                except Exception as e:
                    print(f"[Query] Poetic fallback failed: {e}")

            # As a last-resort local heuristic (no index available), craft a
            # short, careful paraphrase using available sources snippets
            if sources:
                # combine snippets into a short reflective sentence.
                # Clean and filter noisy or boilerplate lines (TOC, copyright,
                # repeated markers like ----- or large whitespace blocks).
                def clean_line(t: str) -> Optional[str]:
                    if not t:
                        return None
                    t = re.sub(r"\s+", " ", t).strip()
                    # skip very short lines or lines that look like table-of-contents
                    if len(t) < 20:
                        return None
                    if re.search(r"page|copyright|chapter|contents|table of", t, re.I):
                        return None
                    # skip lines with repeated non-alphanumeric characters
                    if re.search(r"([^\w\s])\1{4,}", t):
                        return None
                    # if line contains many uppercase words like TOC, skip
                    if sum(1 for w in t.split() if w.isupper()) > 3:
                        return None
                    return t

                cand_parts = []
                for s in sources[:4]:
                    txt = (s.get("snippet") or "").strip()
                    if not txt:
                        continue
                    # Take first 300 chars and split into sentences
                    candidate = txt[:400]
                    sentences = re.split(r"(?<=[.!?])\s+", candidate)
                    for sent in sentences:
                        c = clean_line(sent)
                        if c:
                            cand_parts.append(c)
                            break

                if cand_parts:
                    combined = " — ".join(cand_parts[:3])
                    # normalize whitespace and trim to ~220 chars without cutting words
                    comb = re.sub(r"\s+", " ", combined).strip()
                    if len(comb) > 220:
                        comb = comb[:220]
                        # don't cut mid-word
                        last_space = comb.rfind(" ")
                        if last_space > 100:
                            comb = comb[:last_space]
                        comb = comb.rstrip(".,;:!?") + "…"
                    poetic = f"A poetic reading: {comb}"
                    return poetic, sources

            # If we reach here, prefer an interpretive, source-grounded
            # abstraction rather than a 'no sources' message.
            # 1) If we have an index, ask the poetic/interpretive engine.
            if index:
                try:
                    poetic_engine = index.as_query_engine(
                        similarity_top_k=6,
                        response_mode="compact",
                        text_qa_template=POETIC_ANSWER_PROMPT,
                    )
                    poetic_result = poetic_engine.query(user_query)
                    poetic_answer = str(poetic_result).strip()
                    if poetic_answer and poetic_answer != "I don’t have that in my sources yet.":
                        return poetic_answer, sources
                except Exception as e:
                    print(f"[Query] Poetic engine failed at final fallback: {e}")

            # 2) Build a short abstraction from the collected source snippets.
            def clean_line(t: str) -> Optional[str]:
                if not t:
                    return None
                t = re.sub(r"\s+", " ", t).strip()
                if len(t) < 20:
                    return None
                if re.search(r"page|copyright|chapter|contents|table of", t, re.I):
                    return None
                if re.search(r"([^\w\s])\1{4,}", t):
                    return None
                if sum(1 for w in t.split() if w.isupper()) > 3:
                    return None
                return t

            cand_parts = []
            for s in sources[:6]:
                txt = (s.get("snippet") or "").strip()
                if not txt:
                    continue
                candidate = txt[:500]
                sentences = re.split(r"(?<=[.!?])\s+", candidate)
                for sent in sentences:
                    c = clean_line(sent)
                    if c:
                        cand_parts.append(c)
                        break

            if cand_parts:
                combined = " — ".join(cand_parts[:3])
                comb = re.sub(r"\s+", " ", combined).strip()
                if len(comb) > 240:
                    comb = comb[:240]
                    last_space = comb.rfind(" ")
                    if last_space > 100:
                        comb = comb[:last_space]
                    comb = comb.rstrip(".,;:!?") + "…"
                return (f"A careful reading of the texts suggests: {comb}", sources)

            # 3) As a last resort, attempt to read local files and return a
            # short paragraph-based abstraction (still avoiding error text).
            try:
                src_dir = pathlib.Path("./sources")
                if src_dir.exists():
                    for p in sorted(src_dir.glob("*.txt")):
                        try:
                            text = p.read_text(encoding="utf-8")
                        except Exception:
                            continue
                        paras = [pp.strip() for pp in re.split(r"\n\s*\n", text) if pp.strip()]
                        for para in paras:
                            t = re.sub(r"\s+", " ", para)
                            if len(t) < 120:
                                continue
                            low = t.lower()
                            if any(k in low for k in ("table of contents", "copyright", "acknowledg", "chapter", "introduction", "contents")):
                                continue
                            if re.search(r"([^\w\s])\1{4,}", t):
                                continue
                            if not re.search(r"[\.\!\?]", t):
                                continue
                            sents = re.split(r"(?<=[.!?])\s+", t)
                            chosen = " ".join(sents[:2]).strip()
                            chosen = re.sub(r"\s+", " ", chosen)
                            if len(chosen) > 220:
                                chosen = chosen[:220]
                                last_space = chosen.rfind(" ")
                                if last_space > 100:
                                    chosen = chosen[:last_space]
                                chosen = chosen.rstrip(".,;:!?") + "…"
                            return (f"A careful reading of the texts suggests: {chosen}", sources)
            except Exception:
                pass

            # If everything fails (extremely unlikely with a populated ./sources
            # directory), return a neutral, reflective abstraction.
            return (
                "A careful reading of the texts points toward a reflective, open-ended"
                " stance rather than a categorical claim.",
                sources,
            )

        return answer, sources

    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[Query] Error: {exc}")
        # On unexpected errors return a cautious, source-grounded
        # abstraction rather than a message claiming absence of sources.
        return (
            "A careful reading of the available texts points toward a reflective,"
            " open-ended stance rather than a categorical claim.",
            [],
        )


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
    import sys
    print("[REFRESH] Entering /refresh endpoint", flush=True)
    sys.stdout.flush()
    
    # Optional admin protection: set ORACULAI_ADMIN_TOKEN to require a token
    # provided via query param ?token=... or header X-Admin-Token.
    admin_token = os.environ.get("ORACULAI_ADMIN_TOKEN")
    provided = request.args.get("token") or request.headers.get("X-Admin-Token")
    if admin_token and provided != admin_token:
        print("[REFRESH] Authentication failed", flush=True)
        sys.stdout.flush()
        return "Forbidden", 403
    
    print("[REFRESH] Authentication passed, attempting index build", flush=True)
    sys.stdout.flush()
    
    global index, query_engine
    try:
        print("[REFRESH] Calling _build_index_and_engine()", flush=True)
        sys.stdout.flush()
        index, query_engine = _build_index_and_engine()
        print(f"[REFRESH] Build completed. Index: {index is not None}, Engine: {query_engine is not None}", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"[REFRESH] Exception during build: {type(e).__name__}: {e}", flush=True)
        sys.stdout.flush()
        return f"Index refresh failed with exception: {type(e).__name__}: {e}", 500
    
    if query_engine is None:
        print("[REFRESH] Build succeeded but query_engine is None", flush=True)
        sys.stdout.flush()
        return "Index refresh failed. Check server logs.", 500
    
    print("[REFRESH] Success - returning success response", flush=True)
    sys.stdout.flush()
    return "Index refreshed with new sources."


@app.route("/health")
def health_check():
    """Return a JSON object with the status of the application."""
    global index, query_engine
    # Optionally rebuild the index, but skip by default so health responds fast
    if (query_engine is None or index is None) and os.environ.get(
        "ORACULAI_HEALTH_BUILD", "0"
    ) == "1":
        try:
            built_index, built_engine = _build_index_and_engine()
            if built_engine:
                globals()["index"] = built_index
                globals()["query_engine"] = built_engine
        except Exception as e:
            print(f"[Health] On-demand index build failed: {e}")

    index_is_present = (index is not None) and (not _is_dev_engine(query_engine))
    index_name = os.environ.get("PINECONE_INDEX_NAME") or os.environ.get("PINECONE_INDEX") or "oraculai"
    
    embed_model_name = "unknown"
    if hasattr(Settings, "embed_model") and Settings.embed_model and hasattr(Settings.embed_model, "model_name"):
        embed_model_name = Settings.embed_model.model_name

    llm_model_name = "unknown"
    if hasattr(Settings, "llm") and Settings.llm and hasattr(Settings.llm, "model"):
        llm_model_name = Settings.llm.model

    return {
        "status": "ok",
        "index_present": index_is_present,
        "index_name": index_name,
        "engine_backend": engine_backend,
        "openai_configured": bool(OPENAI_API_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY),
        "embedding_model": embed_model_name,
        "llm_model": llm_model_name,
    }


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
    # Support a single-process, no-reloader launch to make global state
    # (index/query_engine) deterministic. Set ORACULAI_NO_RELOAD=1 to
    # disable the debug reloader when running locally.
    no_reload = os.environ.get("ORACULAI_NO_RELOAD") == "1"
    port = int(os.environ.get("FLASK_RUN_PORT", os.environ.get("PORT", "5001")))
    app.run(debug=(not no_reload), use_reloader=(not no_reload), host="127.0.0.1", port=port)

# create .venv if missing or to start fresh
# python3 -m venv .venv

# activate and install
# source .venv/bin/activate
# python -m pip install --upgrade pip
# pip install -r requirements.txt

# activate .venv first
# To refresh requirements, run this in a shell (not in Python):
# pip freeze | sed '/^-e /d' > requirements.txt
