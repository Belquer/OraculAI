#!/usr/bin/env python3
"""Offline ingestion: read chunk manifest, embed chunks, and upsert to Pinecone.

Usage:
  python scripts/ingest.py --index-name oraculai --batch-size 100 --dry-run

Environment variables:
  PINECONE_API_KEY, PINECONE_ENVIRONMENT, (optional) PINECONE_INDEX_NAME
  EMBEDDING_PROVIDER: 'openai' (default) or 'hf'
  OPENAI_API_KEY (if provider=openai)
  EMBEDDING_MODEL: model name for embeddings (default: text-embedding-3-small)

This script is intentionally defensive: it imports heavy libraries lazily and
prints actionable errors when environment or packages are missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import hashlib
import re
from typing import Iterable, List, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")


def chunks(iterable: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_manifest(manifest_path: pathlib.Path) -> List[Dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Case A: manifest is a dict with top-level 'chunks' list
    if isinstance(data, dict) and "chunks" in data:
        return data["chunks"]

    # Case B: manifest is a list of chunk records
    if isinstance(data, list):
        return data

    # Case C: prepare_chunks.py wrote a mapping of source -> metadata,
    # where each metadata contains a 'chunks_dir' with chunk files.
    if isinstance(data, dict):
        records: List[Dict] = []
        for key, meta in data.items():
            # meta should have 'chunks_dir'
            chunks_dir = meta.get("chunks_dir") if isinstance(meta, dict) else None
            if not chunks_dir:
                continue
            chunks_path = pathlib.Path(chunks_dir)
            if not chunks_path.exists():
                logger.warning("Chunks dir listed in manifest missing: %s", chunks_path)
                continue
            # list chunk files sorted
            files = sorted([p for p in chunks_path.iterdir() if p.is_file()])
            for p in files:
                # store absolute paths so read_chunk_text can open them
                records.append({"path": str(p), "chunk_id": p.name})
        return records

    raise ValueError("Unsupported manifest format: expected list, {'chunks': [...]}, or mapping with 'chunks_dir' entries")


def read_chunk_text(base_dir: pathlib.Path, chunk_rec: Dict) -> str:
    # chunk_rec may be either a string path or dict with 'path'
    path_str = chunk_rec["path"] if isinstance(chunk_rec, dict) else str(chunk_rec)
    path = pathlib.Path(path_str)
    # If path is not absolute, resolve relative to base_dir
    if not path.is_absolute():
        path = (base_dir / path_str).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    return path.read_text(encoding="utf-8")


def get_embeddings_openai(texts: List[str], model: str) -> List[List[float]]:
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package is required for OpenAI embeddings. pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")

    # Support the new OpenAI client (openai>=1.0) preferentially.
    logger.info("Requesting embeddings from OpenAI (model=%s) for %d items", model, len(texts))

    if hasattr(openai, "OpenAI"):
        # Use the new client interface; on HTTP or API errors raise immediately (do not fall back)
        try:
            c = openai.OpenAI(api_key=api_key)
            resp = c.embeddings.create(model=model, input=texts)
        except Exception as e:
            raise RuntimeError(f"OpenAI embeddings request failed: {e}") from e

        # resp.data may be a list of objects or dicts depending on client version
        data_items = getattr(resp, "data", None)
        if data_items is None and isinstance(resp, dict):
            data_items = resp.get("data", [])
        embeddings = []
        for d in data_items:
            # d may be dict-like or object-like
            if isinstance(d, dict):
                emb = d.get("embedding")
            else:
                emb = getattr(d, "embedding", None)
            if emb is None:
                # last resort: try indexing
                try:
                    emb = d["embedding"]
                except Exception:
                    emb = None
            if emb is None:
                raise RuntimeError("Unable to parse embedding from OpenAI response item")
            embeddings.append(emb)
        return embeddings

    # Legacy client path (only used if OpenAI class is absent): set api_key and call openai.Embedding.create
    openai.api_key = api_key
    resp = openai.Embedding.create(model=model, input=texts)
    embeddings = [item["embedding"] for item in resp["data"]]
    return embeddings


def get_embeddings_hf(texts: List[str], model_name: str) -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers is required for hf embeddings. pip install sentence-transformers") from e

    logger.info("Creating HF model %s for %d items", model_name, len(texts))
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def upsert_to_pinecone(index_name: str, vectors: List[Dict]):
    try:
        import pinecone
    except Exception as e:
        raise RuntimeError("pinecone package is required. pip install pinecone-client") from e

    api_key = os.environ.get("PINECONE_API_KEY")
    env = os.environ.get("PINECONE_ENVIRONMENT")
    if not api_key or not env:
        raise RuntimeError("PINECONE_API_KEY and PINECONE_ENVIRONMENT environment variables are required to upsert to Pinecone")

    # sanitize index name: Pinecone requires lower-case alphanum and '-'
    safe_index_name = re.sub(r"[^a-z0-9-]", "-", index_name.lower())
    logger.info("Connecting to Pinecone (index=%s -> %s)", index_name, safe_index_name)

    # Try new Pinecone client API (Pinecone class) first
    if hasattr(pinecone, "Pinecone"):
        try:
            # Create client instance (some versions accept environment kw, some do not)
            try:
                pc = pinecone.Pinecone(api_key=api_key, environment=env)
            except TypeError:
                pc = pinecone.Pinecone(api_key=api_key)

            # list indexes
            try:
                idxs = pc.list_indexes()
                # idxs may provide .names() or be iterable
                if hasattr(idxs, "names"):
                    index_names = list(idxs.names())
                else:
                    index_names = list(idxs)
            except Exception:
                index_names = []

            if safe_index_name not in index_names:
                # Try to create the index automatically if vectors provided
                if not vectors:
                    raise RuntimeError(f"Pinecone index '{index_name}' not found. Create index first or run the index build flow.")
                # infer dimension from first vector values
                first_vals = vectors[0].get("values")
                if not first_vals:
                    raise RuntimeError("Cannot infer embedding dimension from empty vector values; provide index manually or ensure embeddings are computed")
                dim = len(first_vals)
                logger.info("Pinecone index '%s' not found; creating index with dimension=%d", safe_index_name, dim)
                # attempt to create index. Prefer creating without ServerlessSpec first
                created = False
                try:
                    pc.create_index(name=safe_index_name, dimension=dim, metric="cosine")
                    created = True
                except Exception:
                    # try with ServerlessSpec if available and we can supply args
                    ServerlessSpec = getattr(pinecone, "ServerlessSpec", None)
                    if ServerlessSpec:
                        cloud = os.environ.get("PINECONE_CLOUD")
                        region = os.environ.get("PINECONE_REGION")
                        # try to parse PINECONE_ENVIRONMENT like 'us-west1-gcp' -> region='us-west1', cloud='gcp'
                        if not (cloud and region):
                            env_val = os.environ.get("PINECONE_ENVIRONMENT", "")
                            if env_val and "-" in env_val:
                                parts = env_val.split("-")
                                # region := all parts except last joined by '-', cloud := last
                                cloud = parts[-1]
                                region = "-".join(parts[:-1])

                        # Try creating with parsed cloud/region
                        if cloud and region:
                            try:
                                spec = ServerlessSpec(cloud=cloud, region=region)
                                pc.create_index(name=safe_index_name, dimension=dim, metric="cosine", spec=spec)
                                created = True
                            except Exception as ce:
                                logger.warning("ServerlessSpec creation with parsed cloud/region failed: %s", ce)
                                # try a conservative fallback: assume aws and region from env or default
                                try:
                                    fallback_region = region or os.environ.get("PINECONE_ENVIRONMENT", "us-east1")
                                    fallback_cloud = os.environ.get("PINECONE_CLOUD", "aws")
                                    spec = ServerlessSpec(cloud=fallback_cloud, region=fallback_region)
                                    pc.create_index(name=safe_index_name, dimension=dim, metric="cosine", spec=spec)
                                    created = True
                                except Exception as ce2:
                                    raise RuntimeError(f"Failed to create Pinecone index '{index_name}' with ServerlessSpec: {ce2}") from ce2

                if not created:
                    raise RuntimeError(f"Failed to create Pinecone index '{safe_index_name}'. Tried default create and ServerlessSpec paths.")
                # refresh index names
                try:
                    idxs = pc.list_indexes()
                    if hasattr(idxs, "names"):
                        index_names = list(idxs.names())
                    else:
                        index_names = list(idxs)
                except Exception:
                    index_names = []
                if safe_index_name not in index_names:
                    raise RuntimeError(f"Index creation reported success but index '{safe_index_name}' not visible in list_indexes")

            # Get index client
            idx_client = None
            if hasattr(pc, "Index"):
                try:
                    idx_client = pc.Index(safe_index_name)
                except Exception:
                    idx_client = None
            if not idx_client and hasattr(pc, "index"):
                try:
                    idx_client = pc.index(safe_index_name)
                except Exception:
                    idx_client = None

            # Fallback to module-level Index if present
            if not idx_client and hasattr(pinecone, "Index"):
                try:
                    idx_client = pinecone.Index(safe_index_name)
                except Exception:
                    idx_client = None

            if not idx_client:
                raise RuntimeError("Unable to obtain Pinecone index client with the installed pinecone package")

            logger.info("Upserting %d vectors to Pinecone index %s", len(vectors), safe_index_name)
            # Try different upsert signatures
            try:
                idx_client.upsert(vectors=[(v["id"], v["values"], v.get("metadata")) for v in vectors])
                return
            except Exception:
                pass
            try:
                idx_client.upsert(vectors=[{"id": v["id"], "values": v["values"], "metadata": v.get("metadata")} for v in vectors])
                return
            except Exception as e:
                raise RuntimeError(f"Pinecone upsert failed: {e}")

        except Exception as e:
            raise RuntimeError(f"Pinecone client error: {e}") from e

    # Fallback to legacy pinecone.init API
    try:
        pinecone.init(api_key=api_key, environment=env)
        if index_name not in pinecone.list_indexes():
            raise RuntimeError(f"Pinecone index '{index_name}' not found. Create index first or run the index build flow.")
        idx = pinecone.Index(index_name)
        logger.info("Upserting %d vectors to Pinecone index %s (legacy client)", len(vectors), index_name)
        idx.upsert(vectors=[(v["id"], v["values"], v.get("metadata")) for v in vectors])
        return
    except Exception as e:
        raise RuntimeError(f"Pinecone upsert (legacy path) failed: {e}") from e


def build_vector_records(chunk_records: List[Dict], texts: List[str], source_base: pathlib.Path) -> List[Dict]:
    records = []
    for rec, txt in zip(chunk_records, texts):
        # id: sha256(source_path + maybe chunk_id)
        path_str = rec["path"] if isinstance(rec, dict) else str(rec)
        chunk_id = rec.get("chunk_id") if isinstance(rec, dict) else None
        unique = f"{path_str}::{chunk_id}" if chunk_id else path_str
        vid = sha256_hex(unique)
        metadata = {
            "source": path_str,
        }
        if chunk_id:
            metadata["chunk_id"] = chunk_id
        # include short preview
        metadata["preview"] = txt[:300]
        records.append({"id": vid, "values": None, "metadata": metadata})
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="sources_chunks/manifest.json", help="Path to chunk manifest")
    parser.add_argument("--base-dir", default=".", help="Base dir for chunk file paths")
    parser.add_argument("--index-name", default=os.environ.get("PINECONE_INDEX_NAME", "oraculai"), help="Pinecone index name")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of chunks per embedding/upsert batch")
    parser.add_argument("--provider", choices=("openai", "hf"), default=os.environ.get("EMBEDDING_PROVIDER", "openai"), help="Embedding provider")
    parser.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"), help="Embedding model name")
    parser.add_argument("--dry-run", action="store_true", help="Don't upsert, just show counts and samples")

    args = parser.parse_args()

    manifest_path = pathlib.Path(args.manifest)
    base_dir = pathlib.Path(args.base_dir)

    try:
        chunk_recs = load_manifest(manifest_path)
    except Exception as e:
        logger.error("Failed to load manifest: %s", e)
        return 2

    logger.info("Loaded %d chunk records from manifest %s", len(chunk_recs), manifest_path)

    # Build absolute paths list and texts
    paths = [rec["path"] if isinstance(rec, dict) else rec for rec in chunk_recs]
    all_texts = []
    for rec in chunk_recs:
        try:
            txt = read_chunk_text(base_dir, rec)
            all_texts.append(txt)
        except Exception as e:
            logger.warning("Skipping chunk due to read error: %s", e)

    if not all_texts:
        logger.error("No chunk texts available to embed. Exiting.")
        return 3

    logger.info("Preparing to embed %d chunks using provider=%s model=%s", len(all_texts), args.provider, args.model)

    vector_records: List[Dict] = build_vector_records(chunk_recs, all_texts, base_dir)

    # Now batch and embed
    for batch_idx, id_batch in enumerate(chunks(list(range(len(all_texts))), args.batch_size)):
        batch_texts = [all_texts[i] for i in id_batch]
        logger.info("Processing batch %d (%d items)", batch_idx + 1, len(batch_texts))
        try:
            if args.provider == "openai":
                embeddings = get_embeddings_openai(batch_texts, args.model)
            else:
                embeddings = get_embeddings_hf(batch_texts, args.model)
        except Exception as e:
            logger.exception("Embedding failed: %s", e)
            return 4

        # Attach embeddings to vector_records
        for local_idx, global_idx in enumerate(id_batch):
            vector_records[global_idx]["values"] = embeddings[local_idx]

        if args.dry_run:
            logger.info("Dry run: would upsert %d vectors for batch %d", len(batch_texts), batch_idx + 1)
            continue

        # Prepare payload for upsert
        payload = [ {"id": r["id"], "values": r["values"], "metadata": r.get("metadata")} for r in (vector_records[i] for i in id_batch) ]
        try:
            upsert_to_pinecone(args.index_name, payload)
        except Exception as e:
            logger.exception("Upsert failed: %s", e)
            return 5

    logger.info("Ingestion completed (dry_run=%s). Total vectors processed: %d", args.dry_run, sum(1 for r in vector_records if r.get("values") is not None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
