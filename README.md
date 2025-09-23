OraculAI
Spiritual Guidance, Unveiled by AI
Welcome to OraculAI, a browser-based application designed to bring ancient wisdom into the digital age. This platform leverages powerful AI models to act as a personal oracle, providing spiritual guidance and insights from some of the most profound texts in human history.

What It Does
OraculAI is a RAG (Retrieval-Augmented Generation) system that operates on a closed source of spiritual documents. It offers users a unique and deeply personal way to interact with timeless wisdom.

Daily Quote: Every day, the homepage features a new, randomly selected quote from one of our core texts, accompanied by a unique, AI-generated image inspired by the quote itself.

Direct Queries: Users can submit specific questions and receive answers crafted by the AI, drawing exclusively from the provided source material.

Subscription Access: While the daily quote is free, unlimited direct queries are available through a monthly subscription, providing continuous access to your personal oracle.

The Sources
Our initial knowledge base includes three foundational spiritual documents, with plans to expand as the platform grows:

The Emerald Tablet: A cryptic text central to hermetic philosophy.

The Tibetan Book of the Dead: A guide for navigating the intermediate state between death and rebirth.

The I Ching: An ancient Chinese divination text and one of the oldest books in recorded history.

Tech Stack
This application is built with a focus on simplicity, security, and scalability.

Backend: Python, Flask

AI Engine: Notebook LM (or similar models like Llama-Index for closed-source RAG)

Image Generation: DALL-E 3 (or a similar service)

Database: Pinecone

Payments: Stripe

Contribution: We welcome contributions from developers, researchers, and anyone passionate about this project.

Contact: For support, questions, or collaboration inquiries, please contact us at [your email or a contact page link].

## Developer notes

To keep the webserver lean, heavy ML libraries and embedding/model work are done offline.

- Prepare text chunks for ingestion:

```bash
make prepare-chunks
```

- Install runtime dependencies (fast, no heavy ML libs):

```bash
make install
```

- To install developer/ingestion dependencies (transformers, sentence-transformers, torch):

```bash
make dev-install
```

Use `make build-index` or `./.venv/bin/python manage.py build` to run the offline index build (this will import the embedding/vector-store extras when needed).

## Share with testers (no tech skills required)

If you want a simple public URL you can share with non-technical testers, the easiest path is to run the prebuilt container image on Render (or any PaaS that can pull from container registries). Render no longer builds this repo directly because the Docker build stage needs Pinecone/OpenAI secrets; instead GitHub Actions publishes an image to GHCR and Render pulls it.

Render quick start (pulling the prebuilt GHCR image):

1. Ensure the GitHub Action `publish-ghcr` has run on `main`. It produces `ghcr.io/<owner>/oraculai:latest` using `deploy/Dockerfile`.
2. In Render, create a *Web Service* → *Docker* → *Deploy an existing image*.
3. Add a registry credential for GitHub Container Registry (`ghcr.io`) using a GitHub PAT with `read:packages` scope.
4. Set the image to `ghcr.io/<owner>/oraculai:latest` (for this repo: `ghcr.io/belquer/oraculai:latest`).
5. Add the required environment variables:
	- `OPENAI_API_KEY`
	- `PINECONE_API_KEY`
	- `PINECONE_ENVIRONMENT` (e.g. `us-east-1`)
	- `PINECONE_INDEX_NAME` (defaults to `oraculai`)
	- `ORACULAI_SKIP_BUILD=1` (attach to the existing Pinecone index instead of re-ingesting sources)
	- `ORACULAI_NO_RELOAD=1`
6. Deploy. Once the container is up, visit `/health`; if `index_present` is false hit `/refresh` once to warm the Pinecone-backed index.

Tip: the provided `render.yaml` mirrors these settings and is a convenient reference, but Render currently requires you to create the registry credential via the dashboard.

Testing checklist for non-technical testers:

- Open the provided URL in a browser.
- On the homepage read "Today's Synchronicity" and the daily quote.
- Click "Ask" and type a question; press submit.
- Expect a sourced, concise answer. If the system is still warming up you may see a short reflective response.
- If a question fails, copy the URL and report it to the project maintainer.

### GHCR image automation

- `/.github/workflows/publish-ghcr.yml` builds `ghcr.io/<owner>/oraculai:latest` whenever you push to `main`.
- `/.github/workflows/build-image-with-index.yml` (optional) builds the same image with Pinecone/OpenAI secrets injected at build-time so the index is baked inside the container. Run it manually if you need a “warm” image before deploying.

Both workflows accept repository secrets named `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_ENVIRONMENT`. If any are missing the Docker build skips `manage.py build`, so the image will rely on the running service to warm the index via `/refresh`.

Render currently only supports `linux/amd64`, so when building locally use `docker buildx build --platform linux/amd64 ...` (or rely on the workflows above, which already publish that platform).

Make sure repository Actions permissions allow publishing to GHCR (Packages → Write). If you hit permission errors, either enable them in repository settings or provide a PAT through repository secrets.
