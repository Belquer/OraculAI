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

If you want a simple public URL you can share with non-technical testers, the easiest path is to deploy a staging server on Render (or a similar PaaS) and enable auto-deploy from this repository's `staging` or `main` branch.

Quick steps (Render):

1. Create an account at https://render.com and create a new Web Service.
2. Connect your GitHub repository and choose the `main` branch (or `staging` if you prefer).
3. Set the following environment variables in Render's dashboard for the service:
	- `OPENAI_API_KEY` (if you want OpenAI embeddings/completions)
	- `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` (if using Pinecone)
	- `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` (optional)
	- `ORACULAI_NO_RELOAD=1` (recommended for predictable single-process behavior)
4. Render will build the Docker image and provide a public URL (e.g. `https://oraculai-yourname.onrender.com`). Share that URL with testers.

One-click (optional): if you add `RENDER_SERVICE_ID` and `RENDER_API_KEY` as repository secrets, the existing GitHub Actions workflow `/.github/workflows/ci-deploy.yml` can trigger a Render deploy automatically when pushed to `main` or `staging`.

How to get `RENDER_API_KEY` and `RENDER_SERVICE_ID`:

1. In Render, go to Account → API Keys → Create an API Key. Copy the key.
2. In the Render dashboard, open your newly created service and look at the URL; the Service ID is available in the service's Settings → General → Service ID, or you can fetch it via the Render API (`GET https://api.render.com/services`) and find the matching repo name.
3. In your GitHub repository, go to Settings → Secrets → Actions and add two secrets:
	- `RENDER_API_KEY` (the API key from step 1)
	- `RENDER_SERVICE_ID` (the service id from step 2)

Once those secrets are set, pushing to `main` or running the workflow manually will build the image and trigger a Render deploy automatically. The workflow prints the Render response in the Actions logs so you'll see the deploy id and service info.

Testing checklist for non-technical testers:

- Open the provided URL in a browser.
- On the homepage read "Today's Synchronicity" and the daily quote.
- Click "Ask" and type a question; press submit.
- Expect a sourced, concise answer. If the system is still warming up you may see a short reflective response.
- If a question fails, copy the URL and report it to the project maintainer.

If you'd like, I can wire up a Render-specific `render.yaml` and add a Deploy button to this `README.md` so testers can be given a one-click deploy for private instances.

### One-click (Deploy to Render)

You can add a one-click Deploy to Render button so a tester (or you) can create a private instance quickly. Click the button below or paste the YAML into Render when creating a new service.

[![Deploy To Render](https://drive.render.com/buttons/deploy-button-blue.svg)](https://dashboard.render.com/deploy?repo=https://github.com/Belquer/OraculAI)

## Faster deploys: have Render pull a prebuilt image from GHCR

If Render builds are slow, you can configure GitHub Actions to build and push a Docker image to GitHub Container Registry (GHCR), and then change your Render service to pull that image. This reduces deploy times because Render only needs to pull the image instead of building it.

Steps:

1. The included workflow `/.github/workflows/ci-deploy.yml` now contains a `build-and-publish-image` job that pushes `ghcr.io/<owner>/oraculai:latest` on pushes to `main`.
2. In Render, edit your service and switch the deployment method to "Pull from Private Registry" (or equivalent).
	- Registry: ghcr.io
	- Image: `ghcr.io/Belquer/oraculai:latest`
	- Credential: create a Registry Credential in Render (use a GitHub Personal Access Token with `read:packages` scope).
3. Save and deploy — Render will pull the prebuilt image. Subsequent redeploys will be significantly faster.

Note: Make sure repository Actions permissions allow the workflow to publish to GHCR (Package: write). If you see permission errors, enable it in the repository settings or provide a PAT in repository secrets.
