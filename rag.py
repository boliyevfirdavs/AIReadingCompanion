import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

EMBEDDINGS_DIR = Path("embeddings")


def configure_settings() -> None:
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def create_index(book_id: str, documents: list) -> VectorStoreIndex:
    configure_settings()
    splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        show_progress=True,
    )
    index.storage_context.persist(persist_dir=str(EMBEDDINGS_DIR / book_id))
    return index


def load_index(book_id: str) -> VectorStoreIndex:
    configure_settings()
    persist_dir = EMBEDDINGS_DIR / book_id
    if not persist_dir.exists():
        raise FileNotFoundError(f"No index found for book_id '{book_id}'. Run ingest.py first.")
    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage_context)
