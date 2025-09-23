OraculAI — Beta tester quickstart

This file explains how to run a staging instance locally or deploy the Docker image. It assumes you have your own OpenAI and Pinecone keys.

1) Quick local run (recommended for technical testers)

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# create/refresh the index from ./sources
python3 scripts/ingest.py --manifest manifest.json --base-dir . --batch-size 100
export OPENAI_API_KEY="..."
export PINECONE_API_KEY="..."
export PINECONE_INDEX="oraculai"
export ORACULAI_NO_RELOAD=1
export FLASK_APP=app.py
export FLASK_ENV=production
python -m flask run --host=127.0.0.1 --port=5001
```

2) Run via Docker (preferred for non-Python testers)

Build and run locally:

```bash
docker build -t oraculai:staging .
docker run -it --rm -p 5001:5001 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e PINECONE_API_KEY="$PINECONE_API_KEY" \
  -e PINECONE_INDEX="oraculai" \
  oraculai:staging
```

3) Deploy to Render / Fly / Cloud Run (using the GHCR image)

- Run the `publish-ghcr` workflow (or push to `main`) so `ghcr.io/<owner>/oraculai:latest` is available.
- In your platform, create a service that pulls that image (Render: *New Web Service → Docker → Deploy an existing image*).
- Provide a registry credential for GHCR (GitHub PAT with `read:packages`).
- Set environment variables (`OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME=oraculai`, `ORACULAI_NO_RELOAD=1`).
- Expose port 5001 and, after the first deploy, hit `/refresh` if `/health` reports the index is missing.

4) Testing checklist for testers

- Visit the web UI at / and ask several questions.
- Try the /refresh endpoint after adding files to `./sources`.
- Report any incorrect or hallucinated answers as issues using the template below.

5) Feedback link

Please open issues on the repository or use the release discussion link on the release page.
