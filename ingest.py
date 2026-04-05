import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from pypdf import PdfReader

from llama_index.core import Document

from rag import create_index

CATALOG_PATH = Path("books_catalog.json")


def read_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        print(f"Unsupported file type: {suffix}. Supported: .txt, .pdf")
        sys.exit(1)


def load_catalog() -> list[dict]:
    if not CATALOG_PATH.exists():
        return []
    with open(CATALOG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_catalog(catalog: list[dict]) -> None:
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


def ingest_book(file_path: str, book_name: str) -> str:
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"Reading '{path.name}'...")
    text = read_file(path)
    if not text.strip():
        print("File appears to be empty or unreadable.")
        sys.exit(1)

    book_id = uuid.uuid4().hex[:8]
    documents = [Document(text=text, metadata={"book_name": book_name, "book_id": book_id})]

    print(f"Indexing '{book_name}' (chunk_size=1000, overlap=200)...")
    index = create_index(book_id, documents)
    chunk_count = len(index.docstore.docs)

    catalog = load_catalog()
    catalog.append({
        "id": book_id,
        "name": book_name,
        "filename": path.name,
        "chunk_count": chunk_count,
        "created_at": datetime.utcnow().isoformat(),
    })
    save_catalog(catalog)

    print(f"\nDone!")
    print(f"  Name:    {book_name}")
    print(f"  Book ID: {book_id}")
    print(f"  Chunks:  {chunk_count}")
    print(f"\nStart chatting: python chat.py {book_id}")
    return book_id


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest.py <file_path> <book_name>")
        print('Example: python ingest.py books/gatsby.txt "The Great Gatsby"')
        sys.exit(1)

    ingest_book(sys.argv[1], " ".join(sys.argv[2:]))
