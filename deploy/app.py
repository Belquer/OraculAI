"""Deployment entrypoint that reuses the main OraculAI Flask app."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the repository root is on sys.path so we can import the canonical app
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app as canonical_app  # noqa: E402  (import after sys.path tweak)

# Expose the canonical Flask app for WSGI servers (gunicorn, etc.)
app = canonical_app
application = canonical_app


def _run_local() -> None:
    """Run the app in a way that's suitable for container/local use."""
    no_reload = os.environ.get("ORACULAI_NO_RELOAD") == "1"
    port = int(os.environ.get("FLASK_RUN_PORT", os.environ.get("PORT", "5001")))
    canonical_app.run(
        debug=not no_reload,
        use_reloader=not no_reload,
        host=os.environ.get("FLASK_RUN_HOST", "0.0.0.0"),
        port=port,
    )


if __name__ == "__main__":
    _run_local()
