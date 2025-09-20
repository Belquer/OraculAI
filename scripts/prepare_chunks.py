#!/usr/bin/env python3
"""Prepare large chunks for offline ingestion.

This script reads .md and .txt files from ./sources, splits them into
larger chunks controlled by CHUNK_MAX_CHARS and CHUNK_OVERLAP, writes chunk
files to ./sources_chunks/<file_id>/chunk_<n>.txt and writes a manifest.json
with file metadata. This is intentionally lightweight and avoids heavy ML
dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "sources"
CHUNKS = ROOT / "sources_chunks"
MANIFEST = ROOT / "manifest.json"

# Defaults mirror app.py constants but keep local for self-contained script
CHUNK_MAX_CHARS = 4000
CHUNK_OVERLAP = 200


def file_id_for(path: Path) -> str:
    # canonical id: filename without spaces, safe
    return path.name.replace(" ", "_")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str) -> List[str]:
    parts: List[str] = []
    i = 0
    L = len(text)
    while i < L:
        end = i + CHUNK_MAX_CHARS
        part = text[i:end]
        parts.append(part.strip())
        if end >= L:
            break
        i = end - CHUNK_OVERLAP
    return parts


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def prepare() -> int:
    if not SOURCES.exists():
        print("No sources/ folder found.")
        return 1
    CHUNKS.mkdir(exist_ok=True)
    manifest: Dict[str, Dict] = {}
    for p in sorted(SOURCES.glob("*.txt")) + sorted(SOURCES.glob("*.md")):
        fid = file_id_for(p)
        text = read_text(p)
        file_hash = sha256(text)
        chunks = chunk_text(text)
        out_dir = CHUNKS / fid
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, c in enumerate(chunks):
            chunk_path = out_dir / f"chunk_{idx:04d}.txt"
            chunk_path.write_text(c, encoding="utf-8")
        manifest[fid] = {
            "filename": p.name,
            "hash": file_hash,
            "n_chunks": len(chunks),
            "chunks_dir": str(out_dir),
        }
        print(f"Prepared {p.name}: {len(chunks)} chunk(s)")
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(prepare())
