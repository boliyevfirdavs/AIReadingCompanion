# AI Reading Companion

A CLI tool that lets you have conversations with your books. Ingest a PDF or TXT file, then chat with an AI that answers questions grounded in the book's text using RAG (Retrieval-Augmented Generation).

## How it works

1. **Ingest** — a book is parsed, split into chunks, and indexed as embeddings via OpenAI
2. **Chat** — a ReAct agent retrieves relevant passages and answers your questions based on the book content
3. **Persist** — chat history is saved locally so you can resume conversations

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/boliyevfirdavs/AIReadingCompanion.git
cd AIReadingCompanion
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Set your OpenAI API key**

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Usage

### Ingest a book

```bash
python ingest.py <file_path> <book_name>
```

Example:

```bash
python ingest.py books/gatsby.pdf "The Great Gatsby"
```

Supported formats: `.pdf`, `.txt`

This creates an embedding index under `embeddings/` and registers the book in `books_catalog.json`.

### Chat with a book

```bash
# Select or resume a chat
python chat.py <book_id>

# Always start a new chat
python chat.py <book_id> --new

# Resume a specific chat
python chat.py <book_id> <chat_id>
```

Running `python chat.py` with no arguments lists all ingested books and their IDs.

## Project structure

```
.
├── books/              # Put your PDF/TXT books here
├── ingest.py           # Parse and index a book
├── rag.py              # Index creation and loading (LlamaIndex + OpenAI)
├── chat.py             # Interactive chat CLI
├── books_catalog.json  # Registry of ingested books
├── embeddings/         # Auto-generated vector indexes (gitignored)
└── chats/              # Saved chat histories (gitignored)
```

## Models used

| Purpose    | Model                    |
|------------|--------------------------|
| Embeddings | `text-embedding-3-small` |
| Chat / LLM | `gpt-4o-mini`            |