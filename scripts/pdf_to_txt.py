#!/usr/bin/env python3
"""Convert PDFs dropped into ./sources to .txt and optionally delete originals.

Usage:
  python scripts/pdf_to_txt.py         # one-shot convert existing PDFs and delete originals
  python scripts/pdf_to_txt.py --no-delete  # convert but keep originals
  python scripts/pdf_to_txt.py --watch    # watch the sources/ folder and convert new PDFs

The script uses pypdf (already in requirements.txt) to extract text.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

try:
    # pypdf is in requirements.txt
    from pypdf import PdfReader
except Exception as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "Missing dependency 'pypdf'. Install with: pip install pypdf"
    ) from exc


SOURCES = Path("sources")
LOG = logging.getLogger("pdf_to_txt")


def find_pdfs() -> Iterable[Path]:
    if not SOURCES.exists():
        return []
    return sorted(SOURCES.glob("*.pdf"))


def convert_pdf_to_txt(pdf_path: Path, txt_path: Path) -> bool:
    """Extract text from pdf_path and write to txt_path. Return True on success."""
    try:
        reader = PdfReader(str(pdf_path))
        texts = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            texts.append(t)
        content = "\n\n".join([t for t in texts if t.strip()])
        if not content.strip():
            LOG.warning("No text extracted from %s; creating empty file", pdf_path.name)
        txt_path.write_text(content, encoding="utf-8")
        LOG.info("Wrote %s", txt_path)
        return True
    except Exception as exc:
        LOG.exception("Failed to convert %s: %s", pdf_path, exc)
        return False


def process(delete_original: bool = True, archive_dir: Path | None = None) -> int:
    pdfs = list(find_pdfs())
    if not pdfs:
        LOG.info("No PDF files found in %s", SOURCES)
        return 0
    processed = 0
    for pdf in pdfs:
        txt = pdf.with_suffix(".txt")
        LOG.info("Converting %s -> %s", pdf.name, txt.name)
        ok = convert_pdf_to_txt(pdf, txt)
        if ok:
            processed += 1
            if delete_original:
                try:
                    if archive_dir:
                        archive_dir.mkdir(parents=True, exist_ok=True)
                        dest = archive_dir / pdf.name
                        pdf.replace(dest)
                        LOG.info("Moved original %s to %s", pdf.name, dest)
                    else:
                        pdf.unlink()
                        LOG.info("Deleted original %s", pdf.name)
                except Exception:
                    LOG.exception("Failed to remove/archive original %s", pdf)
    return processed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Watch sources/ for new PDFs")
    parser.add_argument("--interval", type=float, default=5.0, help="Poll interval in seconds when watching")
    parser.add_argument("--no-delete", dest="delete", action="store_false", help="Do not delete originals after conversion")
    # By default archive originals to ./archive to preserve them while saving top-level repo space
    parser.add_argument(
        "--archive",
        type=str,
        default="archive",
        help="Move originals to archive directory instead of deleting (default: ./archive)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    archive_dir = Path(args.archive) if args.archive else None

    LOG.info("Starting PDF→TXT processor (delete=%s, archive=%s)", args.delete, archive_dir)

    if args.watch:
        LOG.info("Watch mode enabled. Polling %s every %.1fs", SOURCES, args.interval)
        try:
            while True:
                process(delete_original=args.delete, archive_dir=archive_dir)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            LOG.info("Watch interrupted by user. Exiting.")
            return 0
    else:
        count = process(delete_original=args.delete, archive_dir=archive_dir)
        LOG.info("Processed %d PDF(s)", count)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
