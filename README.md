# OraculAI

Spiritual Guidance, Unveiled by AI

Welcome to OraculAI, a browser-based application designed to bring ancient wisdom into the digital age.  
This platform leverages powerful AI models to act as a personal oracle, providing spiritual guidance and insights from some of the most profound texts in human history.

---

## What It Does

OraculAI is a RAG (Retrieval-Augmented Generation) system that operates on a closed source of spiritual documents.  
It offers users a unique and deeply personal way to interact with timeless wisdom.

- **Daily Quote**: Each visit shows an interpretive reflection grounded in the core texts.  
- **Direct Queries**: Ask specific questions and receive sources-only answers.  
- **Subscription (future)**: Daily quote free; unlimited queries via subscription.

---

## The Sources

Initial knowledge base:
- **The Emerald Tablet** — cryptic hermetic philosophy.
- **The Tibetan Book of the Dead** — guide to the intermediate state.
- **The I Ching** — ancient Chinese divination text.

---

## Tech Stack

- **Backend**: Python (Flask)
- **AI Engine**: LlamaIndex + OpenAI
- **Embeddings**: HuggingFace (MiniLM, 384-dim)
- **Database**: Pinecone
- **UI**: Orb-inspired HTML/CSS
- **Payments (planned)**: Stripe

---

## Getting Started

### Prerequisites
- Python 3.10+
- Pinecone account
- OpenAI API key

### Install
```bash
git clone https://github.com/Belquer/OraculAI.git
cd OraculAI
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt