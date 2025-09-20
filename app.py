import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

load_dotenv()

VERSION = "2.11.0"
APP_PATH = os.path.abspath(__file__)

app = Flask(__name__)

# ==== Retrieval config: LLM, embedder, Pinecone attach (no reingestion) ====
ORACULAI_INDEX_NAME = os.environ.get("ORACULAI_INDEX_NAME", "oraculai-index-384")

# LLM for answering (grounded, low temperature)
Settings.llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0.0, max_tokens=220)

# 384-dim embedder to match your index
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

_index = None  # will be attached lazily

def _attach_index():
    """Attach to existing Pinecone index without re-ingesting documents."""
    from pinecone import Pinecone   # local import to avoid import-time failures
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pine_idx = pc.Index(ORACULAI_INDEX_NAME)

    vs = PineconeVectorStore(pinecone_index=pine_idx, namespace="")
    # NOTE: from_vector_store attaches to existing vectors only.
    return VectorStoreIndex.from_vector_store(vs)

@app.route("/refresh")
def refresh():
    """(Re)attach to the Pinecone index; does NOT rebuild or upsert."""
    global _index
    try:
        _index = _attach_index()
        return "Index attached."
    except Exception as e:
        print("[/refresh] attach error:", e)
        return "Failed to attach index.", 500
# ========================================================================== 


QA_PROMPT = """You are a wise and poetic guide. Answer the question with gentle eloquence using ONLY the provided context as your factual foundation.
- You may synthesize themes clearly supported by the context, but do not import outside facts or terminology not present in the context (e.g., avoid “soul” unless it appears).
- If the context is unrelated, say exactly: "I don’t have that in my sources yet."
- Keep the answer under ~120 words and end with a complete sentence.

Context:
{context_str}

Question:
{query_str}

Answer:"""


def _expand_query(q: str) -> str:
    """Very light query expansion to catch source synonyms/phrases."""
    ql = q.lower()
    extras = []

    # direct 'bardo' support
    if "bardo" in ql:
        extras += [
            "intermediate state",
            "between death and rebirth",
            "after death",
            "rebirth",
            "liberation at the time of death",
        ]

    # afterlife / bardo / rebirth mappings
    if "afterlife" in ql or "life after death" in ql or "after death" in ql:
        extras += [
            "bardo",
            "intermediate state",
            "between death and rebirth",
            "rebirth",
            "liberation at the time of death",
        ]

    if "soul" in ql or "spirit" in ql:
        extras += [
            # core Buddhist renderings that may appear in translations
            "mindstream", "mind stream", "mental continuum", "continuity of consciousness",
            "consciousness", "pristine cognition", "alaya", "storehouse consciousness",
            # bridge terms often contrasted with “soul”
            "self", "no-self", "anatman", "anatta", "ego", "mind", "awareness", "nature of mind"
        ]

    if "reincarn" in ql or "rebirth" in ql:
        extras += ["karma", "cause and effect"]

    if "destiny" in ql or "fate" in ql:
        extras += ["change", "transformation"]

    if not extras:
        return q

    # dedupe while preserving order
    return q + ". " + ". ".join(dict.fromkeys(extras))


@app.route("/")
def index():
    return render_template("index.html", quote="(test) template wiring OK")

@app.route("/version")
def version():
    return f"OraculAI v{VERSION} · {APP_PATH}"

@app.route("/query", methods=["POST"])
def query():
    from flask import render_template  # local import to be explicit
    global _index

    question = (request.form.get("user_query") or "").strip()
    if not question:
        return render_template("response.html", query="", response="Please enter a question.")

    # Ensure we’re attached to the existing Pinecone index (no re-ingestion)
    try:
        if _index is None:
            _index = _attach_index()
    except Exception as e:
        print("[/query] attach error:", e)
        return render_template("response.html", query=question, response="I don’t have that in my sources yet.")

    try:
        # 1) Expand only for retrieval (synonyms like bardo/intermediate state)
        expanded = _expand_query(question)

        # 2) Retrieve top chunks
        retriever = _index.as_retriever(similarity_top_k=8)
        nodes = retriever.retrieve(expanded) or []

        # 3) Collect clean text
        chunks = []
        for n in nodes:
            txt = n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
            if txt:
                chunks.append(txt.strip())

        # 4) If nothing retrieved → no hallucinations
        if not chunks:
            return render_template("response.html", query=question, response="I don’t have that in my sources yet.")

        # 5) Build grounded context and ask with strict QA prompt
        #    Use ORIGINAL question for answering (poetic but grounded)
        context_str = "\n\n---\n\n".join(c[:500] for c in chunks)[:2000]
        prompt = QA_PROMPT.format(query_str=question, context_str=context_str)

        resp = Settings.llm.complete(prompt)
        answer = (str(resp) or "").strip() or "I don’t have that in my sources yet."

        return render_template("response.html", query=question, response=answer)

    except Exception as e:
        print("[/query] error:", e)
        return render_template("response.html", query=question, response="I don’t have that in my sources yet.")