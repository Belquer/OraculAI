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
