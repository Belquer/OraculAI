"""Management CLI for OraculAI: build or refresh vector index without starting the webserver."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from typing import Optional

APP_PY = Path(__file__).with_name("app.py")


def run_build():
    """Import app and call its _build_index_and_engine() function.

    This runs outside the webserver so heavy imports and long-running
    operations are explicit and easy to track.
    """
    env = os.environ
    sys.path.insert(0, str(Path(__file__).parent))
    import app

    print("Building index (this may take a while)...")
    index, engine = app._build_index_and_engine()
    if engine is None:
        print("Index build failed. Check logs for details.")
        return 1
    print("Index built successfully.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    print(f"manage.py received arguments: {sys.argv}")
    if not argv:
        print("Usage: python manage.py build")
        return 1

    cmd = argv[0].strip().lower()
    if cmd == "build" or cmd == "refresh":
        return run_build()

    print(f"Unknown command: '{cmd}'")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
