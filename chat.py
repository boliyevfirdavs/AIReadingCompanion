import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool, QueryEngineTool

from rag import load_index

CATALOG_PATH = Path("books_catalog.json")
CHATS_DIR = Path("chats")


def find_book(book_id: str) -> dict | None:
    if not CATALOG_PATH.exists():
        return None
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    for book in catalog:
        if book["id"] == book_id:
            return book
    return None


def chat_path(book_id: str, chat_id: str) -> Path:
    return CHATS_DIR / book_id / f"{chat_id}.json"


def load_chat(book_id: str, chat_id: str) -> dict | None:
    path = chat_path(book_id, chat_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_chat(book_id: str, chat_id: str, session: dict) -> None:
    path = chat_path(book_id, chat_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)


def list_chats(book_id: str) -> list[dict]:
    book_chats_dir = CHATS_DIR / book_id
    if not book_chats_dir.exists():
        return []
    sessions = []
    for p in sorted(book_chats_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        sessions.append({
            "chat_id": p.stem,
            "title": data.get("title", "Untitled"),
            "updated_at": data.get("updated_at", ""),
            "message_count": len([m for m in data.get("messages", []) if m["role"] == "user"]),
        })
    return sessions


def make_title(messages: list[dict]) -> str:
    for m in messages:
        if m["role"] == "user" and isinstance(m.get("content"), str):
            text = m["content"].strip()
            return text[:60] + ("..." if len(text) > 60 else "")
    return "Untitled"


def run_chat(book_id: str, chat_id: str, session: dict) -> None:
    book = session["book"]
    saved_messages = session.get("messages", [])

    print(f"\nReading Companion — {book['name']}")
    print(f"Chat ID: {chat_id} | Type 'quit' to exit\n")

    prior_count = len([m for m in saved_messages if m["role"] == "user"])
    if prior_count > 0:
        print(f"[Resuming chat — {prior_count} previous message(s)]\n")

    # Restore conversation history into LlamaIndex memory
    chat_history = [
        ChatMessage(role=MessageRole(m["role"]), content=m["content"])
        for m in saved_messages
        if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    memory = ChatMemoryBuffer.from_defaults(chat_history=chat_history, token_limit=4000)

    index = load_index(book_id)
    query_engine = index.as_query_engine(similarity_top_k=5)
    search_tool = QueryEngineTool.from_defaults(
        query_engine,
        description=f"Search passages from '{book['name']}' to answer questions about the book.",
    )

    nodes_by_position = sorted(
        index.docstore.docs.values(),
        key=lambda n: n.end_char_idx if n.end_char_idx is not None else 0,
    )

    def get_book_ending() -> str:
        return nodes_by_position[-1].text if nodes_by_position else ""

    def get_book_opening() -> str:
        return nodes_by_position[0].text if nodes_by_position else ""

    ending_tool = FunctionTool.from_defaults(fn=get_book_ending)
    opening_tool = FunctionTool.from_defaults(fn=get_book_opening)

    agent = ReActAgent(
        tools=[search_tool, ending_tool, opening_tool],
        system_prompt=(
            f"You are a helpful reading companion for the book '{book['name']}'. "
            "Always use the search tool to find relevant passages before answering. "
            "Base your answers on the retrieved passages. If passages don't contain "
            "enough information, say so honestly."
        ),
        verbose=False,
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        async def _run():
            return await agent.run(user_msg=user_input, memory=memory)
        response = asyncio.run(_run())
        print(f"\nAssistant: {response}\n")

        # Persist user + assistant messages only
        session["messages"] = [
            {"role": m.role.value, "content": m.content}
            for m in memory.get_all()
            if m.role.value in ("user", "assistant") and m.content
        ]
        if session.get("title") == "Untitled":
            session["title"] = make_title(session["messages"])
        save_chat(book_id, chat_id, session)


def prompt_chat_selection(book_id: str, book: dict) -> tuple[str, dict]:
    existing = list_chats(book_id)

    if existing:
        print(f"\nBook: {book['name']}\n")
        print("  [0] New chat")
        for i, s in enumerate(existing, 1):
            print(f"  [{i}] {s['title']}  ({s['message_count']} messages, {s['updated_at'][:10]})")
        print()
        choice = input("Select a chat or press Enter for new: ").strip()
        if choice == "" or choice == "0":
            return _new_session(book_id, book)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(existing):
                chat_id = existing[idx]["chat_id"]
                session = load_chat(book_id, chat_id)
                return chat_id, session
        except ValueError:
            pass
        print("Invalid choice, starting new chat.")

    return _new_session(book_id, book)


def _new_session(book_id: str, book: dict) -> tuple[str, dict]:
    chat_id = uuid.uuid4().hex[:8]
    session = {
        "chat_id": chat_id,
        "book_id": book_id,
        "book": book,
        "title": "Untitled",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    }
    return chat_id, session


def chat(book_id: str, chat_id: str | None = None, new: bool = False) -> None:
    book = find_book(book_id)
    if not book:
        print(f"Book ID '{book_id}' not found in catalog. Run ingest.py first.")
        sys.exit(1)

    if chat_id:
        session = load_chat(book_id, chat_id)
        if not session:
            print(f"Chat ID '{chat_id}' not found.")
            sys.exit(1)
    elif new:
        chat_id, session = _new_session(book_id, book)
    else:
        chat_id, session = prompt_chat_selection(book_id, book)

    run_chat(book_id, chat_id, session)


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python chat.py <book_id>               — select or start a chat")
        print("  python chat.py <book_id> --new         — always start a new chat")
        print("  python chat.py <book_id> <chat_id>     — resume a specific chat")
        print("\nAvailable books:")
        if CATALOG_PATH.exists():
            with open(CATALOG_PATH, encoding="utf-8") as f:
                catalog = json.load(f)
            if catalog:
                for book in catalog:
                    print(f"  {book['id']}  {book['name']}")
            else:
                print("  (none — run ingest.py to add a book)")
        sys.exit(1)

    book_id = args[0]
    if len(args) >= 2 and args[1] == "--new":
        chat(book_id, new=True)
    elif len(args) >= 2:
        chat(book_id, chat_id=args[1])
    else:
        chat(book_id)
