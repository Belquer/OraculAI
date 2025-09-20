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

3) Deploy to Render / Fly / Cloud Run

- Create a new service and point it at this repository and the `staging` branch.
- Set environment variables (OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, ORACULAI_NO_RELOAD=1).
- Configure the service to expose port 5001 and health checks.

4) Testing checklist for testers

- Visit the web UI at / and ask several questions.
- Try the /refresh endpoint after adding files to `./sources`.
- Report any incorrect or hallucinated answers as issues using the template below.

5) Feedback link

Please open issues on the repository or use the release discussion link on the release page.

