"""Lightweight copy of app.py for deploy context."""

from __future__ import annotations

import os
from flask import Flask, render_template

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


@app.route("/health")
def health():
    return {"server": "ok", "index_present": False}


@app.route("/")
def index():
    quote = "A small collection of reflective prompts for OraculAI"
    try:
        with open("quotes.txt", "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            if lines:
                quote = lines[0]
    except Exception:
        pass
    return render_template("index.html", quote=quote)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
