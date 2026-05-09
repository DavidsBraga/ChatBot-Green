"""
Gradio Web Interface for EcoGuide Chatbot.

This app connects the authenticated frontend to the existing RAG backend:
- Azure OpenAI LLM and embeddings
- FAISS retrieval with source metadata
- CSV sustainable product recommendations
- Local JSON login/profile/conversation storage
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import huggingface_hub

    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @staticmethod
            def get_token():
                return huggingface_hub.get_token()

            @staticmethod
            def save_token(token):
                return None

            @staticmethod
            def delete_token():
                return None

        huggingface_hub.HfFolder = HfFolder
except ImportError:
    pass

import gradio as gr
import gradio_client.utils as gradio_client_utils


_original_get_type = gradio_client_utils.get_type


def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return _original_get_type(schema)


gradio_client_utils.get_type = _safe_get_type


APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)


def load_app_environment() -> None:
    env_candidates = [
        APP_DIR / ".env",
        APP_DIR / "gen-ai-business-case-recruiting" / ".env",
        Path.home() / "Desktop" / "gen-ai-business-case-recruiting" / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            return

    load_dotenv(override=True)


load_app_environment()

USERS_FILE = APP_DIR / "users.json"
CONVERSATIONS_FILE = APP_DIR / "conversations.json"
EXPORTS_DIR = APP_DIR / "exports"
DEMO_EMAIL = "demo@ecoguide.test"
DEMO_PASSWORD = "demo123"
DEMO_NAME = "EcoGuide Demo User"

PRIMARY_GREEN = "#86BC25"
DARK_GREEN = "#0B3D2E"
LIGHT_BG = "#F6FFF2"

_llm = None
_index = None
_csv_loader = None
_rag_ready = False
APP_SETTINGS = {
    "num_chunks": 5,
    "show_sources": True,
    "temperature": 0.7,
}

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".html", ".htm", ".docx", ".pptx", ".csv"}
UPLOADED_DOCUMENTS_DIR = APP_DIR / "uploaded_documents"
FAISS_INDEX_DIR = APP_DIR / "faiss_index"


CUSTOM_CSS = f"""
:root {{
  --eco-green: {PRIMARY_GREEN};
  --eco-dark: {DARK_GREEN};
  --eco-bg: {LIGHT_BG};
}}

@keyframes ecoFadeIn {{
  from {{
    opacity: 0;
    transform: translateY(12px);
  }}
  to {{
    opacity: 1;
    transform: translateY(0);
  }}
}}

@keyframes ecoButtonGlow {{
  0%, 100% {{
    box-shadow: 0 8px 20px rgba(11, 61, 46, 0.14);
  }}
  50% {{
    box-shadow: 0 12px 30px rgba(134, 188, 37, 0.42);
  }}
}}

@keyframes ecoButtonBreath {{
  0%, 100% {{
    transform: translateY(0) scale(1);
  }}
  50% {{
    transform: translateY(-1px) scale(1.015);
  }}
}}

.gradio-container {{
  background: var(--eco-bg) !important;
  color: #10231b;
  font-family: Inter, Arial, sans-serif;
}}

#app-shell {{
  max-width: 1180px;
  margin: 0 auto;
}}

#landing-card, #auth-card, #profile-card {{
  background: #ffffff;
  border: 1px solid rgba(11, 61, 46, 0.12);
  border-radius: 8px;
  box-shadow: 0 18px 48px rgba(11, 61, 46, 0.12);
  padding: 34px;
  animation: ecoFadeIn 420ms ease-out both;
}}

#landing-brand {{
  color: var(--eco-dark);
  font-size: 56px;
  line-height: 1;
  font-weight: 800;
  letter-spacing: 0;
  margin-bottom: 10px;
}}

#landing-brand span,
#header-logo span {{
  color: var(--eco-green);
}}

#landing-subtitle {{
  color: #2f4f43;
  font-size: 20px;
  margin-bottom: 28px;
}}

#app-header {{
  background: #ffffff;
  border: 1px solid rgba(11, 61, 46, 0.12);
  border-radius: 8px;
  box-shadow: 0 10px 30px rgba(11, 61, 46, 0.10);
  padding: 12px 16px;
  margin-bottom: 14px;
  align-items: center;
  animation: ecoFadeIn 360ms ease-out both;
}}

#header-logo {{
  color: var(--eco-dark);
  font-size: 30px;
  line-height: 1;
  font-weight: 800;
  letter-spacing: 0;
  white-space: nowrap;
}}

#header-actions {{
  justify-content: flex-end;
  gap: 10px;
}}

#sidebar {{
  background: var(--eco-dark);
  border-radius: 8px;
  padding: 18px;
  min-height: 650px;
  animation: ecoFadeIn 420ms ease-out both;
}}

#sidebar * {{
  color: #ffffff;
}}

#sidebar input, #sidebar textarea, #sidebar select {{
  color: #10231b !important;
}}

[role="listbox"],
.gradio-container [role="listbox"],
#sidebar [role="listbox"] {{
  background: #ffffff !important;
  border: 1px solid rgba(11, 61, 46, 0.18) !important;
  box-shadow: 0 16px 34px rgba(11, 61, 46, 0.16) !important;
}}

[role="listbox"] *,
[role="option"],
[role="option"] *,
.gradio-container [role="listbox"] *,
.gradio-container [role="option"],
.gradio-container [role="option"] *,
#sidebar [role="listbox"] *,
#sidebar [role="option"],
#sidebar [role="option"] * {{
  color: #10231b !important;
}}

[role="option"]:hover,
[role="option"][aria-selected="true"],
.gradio-container [role="option"]:hover,
.gradio-container [role="option"][aria-selected="true"] {{
  background: #eef8df !important;
  color: #10231b !important;
}}

#chat-card {{
  background: #ffffff;
  border: 1px solid rgba(11, 61, 46, 0.10);
  border-radius: 8px;
  padding: 18px;
  min-height: 650px;
  box-shadow: 0 14px 38px rgba(11, 61, 46, 0.10);
  animation: ecoFadeIn 480ms ease-out both;
}}

#chat-title {{
  color: var(--eco-dark);
  font-size: 30px;
  line-height: 1.2;
  font-weight: 760;
  letter-spacing: 0;
  margin: 0 0 8px 0;
}}

#profile-card {{
  max-width: 760px;
  width: 100%;
  margin: 0 auto;
}}

#theme-bar {{
  display: flex;
  justify-content: flex-end;
  align-items: center;
  padding: 14px 0 10px 0;
}}

.primary-action button,
button.primary-action,
.gradio-container .primary-action button {{
  background: var(--eco-green) !important;
  border-color: var(--eco-green) !important;
  color: #0b1f16 !important;
  font-weight: 700 !important;
  animation: ecoButtonGlow 2.7s ease-in-out infinite, ecoButtonBreath 2.7s ease-in-out infinite;
}}

.secondary-action button,
button.secondary-action,
.gradio-container .secondary-action button {{
  background: #ffffff !important;
  border-color: rgba(11, 61, 46, 0.25) !important;
  color: var(--eco-dark) !important;
  font-weight: 650 !important;
}}

.danger-action button,
button.danger-action,
.gradio-container .danger-action button {{
  background: #f7d9d9 !important;
  border-color: #e7aaaa !important;
  color: #6d1515 !important;
  font-weight: 650 !important;
}}

button,
.gradio-container button {{
  border-radius: 8px !important;
  transform: translateY(0) scale(1);
  position: relative;
  overflow: hidden;
  transition:
    transform 190ms cubic-bezier(0.2, 0.8, 0.2, 1),
    box-shadow 190ms ease,
    filter 190ms ease,
    border-color 190ms ease,
    background-color 190ms ease !important;
  will-change: transform, box-shadow;
}}

button::after,
.gradio-container button::after {{
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(110deg, transparent 20%, rgba(255, 255, 255, 0.34) 46%, transparent 72%);
  transform: translateX(-120%);
  transition: transform 420ms ease;
  pointer-events: none;
}}

button:hover,
.gradio-container button:hover {{
  transform: translateY(-4px) scale(1.025) !important;
  box-shadow: 0 16px 34px rgba(11, 61, 46, 0.22) !important;
  filter: brightness(1.06);
}}

button:hover::after,
.gradio-container button:hover::after {{
  transform: translateX(120%);
}}

button:active,
.gradio-container button:active {{
  transform: translateY(0) scale(0.96) !important;
  box-shadow: 0 5px 14px rgba(11, 61, 46, 0.12) !important;
  filter: brightness(0.98);
}}

button:focus-visible,
.gradio-container button:focus-visible {{
  outline: 3px solid rgba(134, 188, 37, 0.42) !important;
  outline-offset: 2px !important;
}}

button:disabled,
button[disabled],
.gradio-container button:disabled,
.gradio-container button[disabled],
button[aria-disabled="true"],
.gradio-container button[aria-disabled="true"] {{
  transform: none !important;
  box-shadow: none !important;
  filter: grayscale(0.15) opacity(0.72);
  cursor: not-allowed !important;
  animation: none !important;
}}

.secondary-action button:hover,
.gradio-container .secondary-action button:hover,
button.secondary-action:hover {{
  border-color: rgba(134, 188, 37, 0.55) !important;
}}

.danger-action button:hover,
.gradio-container .danger-action button:hover,
button.danger-action:hover {{
  box-shadow: 0 12px 26px rgba(109, 21, 21, 0.18) !important;
  filter: brightness(1.01);
}}

textarea, input {{
  border-radius: 8px !important;
}}

textarea:focus, input:focus {{
  border-color: var(--eco-green) !important;
  box-shadow: 0 0 0 3px rgba(134, 188, 37, 0.16) !important;
}}

.status-line {{
  color: #24483b;
  font-size: 14px;
}}

#sidebar .status-line {{
  color: #dbeed4;
}}

@media (max-width: 820px) {{
  #landing-brand {{
    font-size: 42px;
  }}
  #header-logo {{
    font-size: 25px;
  }}
  #header-actions {{
    justify-content: flex-start;
  }}
  #sidebar, #chat-card {{
    min-height: auto;
  }}
}}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_users() -> dict[str, dict[str, Any]]:
    return read_json(USERS_FILE, {})


def save_users(users: dict[str, dict[str, Any]]) -> None:
    write_json(USERS_FILE, users)


def load_conversations() -> list[dict[str, Any]]:
    return read_json(CONVERSATIONS_FILE, [])


def save_conversations(conversations: list[dict[str, Any]]) -> None:
    write_json(CONVERSATIONS_FILE, conversations)


def normalize_email(email: str | None) -> str:
    return (email or "").strip().lower()


def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


def verify_password(password: str, user: dict[str, Any]) -> bool:
    return hash_password(password, user.get("salt", "")) == user.get("password_hash")


def ensure_demo_user() -> None:
    users = load_users()
    salt = users.get(DEMO_EMAIL, {}).get("salt") or secrets.token_hex(16)
    users[DEMO_EMAIL] = {
        "name": DEMO_NAME,
        "email": DEMO_EMAIL,
        "salt": salt,
        "password_hash": hash_password(DEMO_PASSWORD, salt),
        "confirmed": True,
        "created_at": users.get(DEMO_EMAIL, {}).get("created_at", utc_now()),
        "confirmed_at": users.get(DEMO_EMAIL, {}).get("confirmed_at", utc_now()),
        "demo": True,
    }
    save_users(users)


def title_from_message(message: str) -> str:
    cleaned = " ".join((message or "").strip().split())
    if not cleaned:
        return "New chat"
    return cleaned[:42] + ("..." if len(cleaned) > 42 else "")


def get_user_conversations(user_email: str | None) -> list[dict[str, Any]]:
    email = normalize_email(user_email)
    conversations = [
        item for item in load_conversations()
        if normalize_email(item.get("user_email")) == email
    ]
    return sorted(conversations, key=lambda item: item.get("updated_at", ""), reverse=True)


def get_conversation(conversation_id: str | None, user_email: str | None) -> dict[str, Any] | None:
    email = normalize_email(user_email)
    for conversation in load_conversations():
        if conversation.get("id") == conversation_id and normalize_email(conversation.get("user_email")) == email:
            return conversation
    return None


def conversation_choices(user_email: str | None) -> list[tuple[str, str]]:
    return [
        (conversation.get("title") or "New chat", conversation["id"])
        for conversation in get_user_conversations(user_email)
    ]


def conversation_dropdown_update(user_email: str | None, selected_id: str | None = None):
    choices = conversation_choices(user_email)
    valid_ids = {value for _, value in choices}
    value = selected_id if selected_id in valid_ids else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value)


def create_conversation(user_email: str, title: str = "New chat") -> dict[str, Any]:
    now = utc_now()
    conversation = {
        "id": str(uuid.uuid4()),
        "user_email": normalize_email(user_email),
        "title": title.strip() or "New chat",
        "messages": [],
        "created_at": now,
        "updated_at": now,
    }
    conversations = load_conversations()
    conversations.append(conversation)
    save_conversations(conversations)
    return conversation


def update_conversation(conversation: dict[str, Any]) -> None:
    conversations = load_conversations()
    for index, item in enumerate(conversations):
        if item.get("id") == conversation.get("id"):
            conversations[index] = conversation
            save_conversations(conversations)
            return
    conversations.append(conversation)
    save_conversations(conversations)


def uploaded_document_choices() -> list[tuple[str, str]]:
    if not UPLOADED_DOCUMENTS_DIR.exists():
        return []
    return [
        (path.name, path.name)
        for path in sorted(UPLOADED_DOCUMENTS_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    ]


def uploaded_document_dropdown_update(selected_name: str | None = None):
    choices = uploaded_document_choices()
    valid_names = {value for _, value in choices}
    value = selected_name if selected_name in valid_names else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value)


def safe_uploaded_filename(filename: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in " ._-" else "_" for char in filename)
    cleaned = cleaned.strip(" ._")
    return cleaned or f"document_{uuid.uuid4().hex}"


def make_unique_upload_path(filename: str) -> Path:
    UPLOADED_DOCUMENTS_DIR.mkdir(exist_ok=True)
    safe_name = safe_uploaded_filename(filename)
    candidate = UPLOADED_DOCUMENTS_DIR / safe_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    for counter in range(2, 1000):
        next_candidate = UPLOADED_DOCUMENTS_DIR / f"{stem}_{counter}{suffix}"
        if not next_candidate.exists():
            return next_candidate

    return UPLOADED_DOCUMENTS_DIR / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def ingest_document_file(index, path: Path) -> None:
    from src.ingestion.loaders.loader import Loader

    extension = path.suffix.lower().lstrip(".")
    if extension == "csv":
        return

    loader = Loader(extension=extension, filepath=str(path))
    if extension == "pdf":
        for page_num, text in loader.loader.extract_text_by_page():
            index.ingest_text(text=text, metadata={"source": path.name, "page": page_num})
        return

    text = loader.extract_text()
    index.ingest_text(text=text, metadata={"source": path.name})


def ingest_uploaded_documents(index) -> None:
    for _, filename in uploaded_document_choices():
        ingest_document_file(index, UPLOADED_DOCUMENTS_DIR / filename)


def reset_rag_index_after_document_change() -> None:
    global _llm, _index, _rag_ready

    _llm = None
    _index = None
    _rag_ready = False
    if FAISS_INDEX_DIR.exists():
        shutil.rmtree(FAISS_INDEX_DIR)


def initialize_rag_once() -> None:
    global _llm, _index, _csv_loader, _rag_ready

    if _rag_ready:
        return

    from src.ingestion.ingest_files import ingest_files_data_folder
    from src.ingestion.loaders.loaderCSV import LoaderCSV
    from src.services.models.embeddings import Embeddings
    from src.services.models.llm import LLM
    from src.services.vectorial_db.faiss_index import FAISSIndex

    _llm = LLM()
    embeddings = Embeddings()
    _index = FAISSIndex(embeddings=embeddings.get_embeddings, dimension=3072)

    try:
        _index.load_index()
    except FileNotFoundError:
        ingest_files_data_folder(_index)
        ingest_uploaded_documents(_index)
        _index.save_index()

    products_path = APP_DIR / "data" / "sustainable_products.csv"
    if products_path.exists():
        _csv_loader = LoaderCSV(str(products_path))

    _rag_ready = True


def history_to_llm_messages(chat_history: list[Any] | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in chat_history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user_msg, bot_msg = item[0], item[1]
            if user_msg:
                messages.append({"role": "user", "content": str(user_msg)})
            if bot_msg:
                messages.append({"role": "assistant", "content": str(bot_msg)})
    return messages


def normalize_retrieved_chunk(item: Any, rank: int) -> dict[str, Any]:
    if isinstance(item, dict):
        chunk = item.get("chunk") or item.get("text") or ""
        metadata = item.get("metadata") or {}
        score = item.get("score")
        return {"chunk": str(chunk), "metadata": metadata, "score": score}
    return {"chunk": str(item), "metadata": {"chunk_index": rank}, "score": None}


def format_sources(retrieved_chunks: list[dict[str, Any]]) -> str:
    if not APP_SETTINGS.get("show_sources", True):
        return ""

    seen = set()
    lines = ["**Sources:**"]
    for item in retrieved_chunks:
        metadata = item.get("metadata") or {}
        source = metadata.get("source") or metadata.get("file_name") or "Knowledge base"
        page = metadata.get("page")
        label = f"{source}, page {page}" if page else str(source)
        if label not in seen:
            seen.add(label)
            lines.append(f"- {label}")

    if len(lines) == 1:
        lines.append("- Retrieved from the indexed FAISS knowledge base.")

    return "\n".join(lines)


def should_recommend_products(message: str) -> bool:
    keywords = [
        "recommend", "recommendation", "product", "products", "buy", "shop",
        "suggest", "sustainable product", "eco product", "home", "where can i get",
    ]
    lowered = (message or "").lower()
    return any(keyword in lowered for keyword in keywords)


def format_product_recommendations(message: str) -> str:
    if not _csv_loader or not should_recommend_products(message):
        return ""

    products = _csv_loader.search_products(message, top_n=3)
    if not products:
        return ""

    lines = ["**Recommended sustainable products:**"]
    for product in products:
        lines.extend(
            [
                "",
                f"**{product.get('Product Name', 'Product')}**",
                f"- Category: {product.get('Category', 'N/A')}",
                f"- Price: ${product.get('Price ($)', 'N/A')}",
                f"- Review score: {product.get('Review Score', 'N/A')}",
                f"- Ships to: {product.get('Countries', 'N/A')}",
                f"- {product.get('Description', '')}",
            ]
        )
    return "\n".join(lines)


def rag_answer(message: str, chat_history: list[Any] | None) -> str:
    initialize_rag_once()
    llm_history = history_to_llm_messages(chat_history)
    num_chunks = int(APP_SETTINGS.get("num_chunks", 5))
    raw_chunks = _index.retrieve_chunks(message, num_chunks=num_chunks)
    retrieved_chunks = [normalize_retrieved_chunk(item, rank) for rank, item in enumerate(raw_chunks)]

    context_parts = []
    for item in retrieved_chunks:
        metadata = item.get("metadata") or {}
        source = metadata.get("source") or metadata.get("file_name") or "Knowledge base"
        page = metadata.get("page")
        label = f"{source}, page {page}" if page else source
        context_parts.append(f"[Source: {label}]\n{item.get('chunk', '')}")

    context = "\n\n#####\n\n".join(context_parts)
    response = _llm.get_response(llm_history, context, message)

    sources = format_sources(retrieved_chunks)
    products = format_product_recommendations(message)
    extras = [part for part in (sources, products) if part]
    if extras:
        response = f"{response}\n\n---\n" + "\n\n".join(extras)
    return response


def list_indexed_documents() -> str:
    lines = ["### Current documents"]
    for folder, label in ((APP_DIR / "data", "Data folder"), (UPLOADED_DOCUMENTS_DIR, "Uploaded for all chats")):
        if not folder.exists():
            continue
        files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS)
        if files:
            lines.append(f"\n**{label}**")
            lines.extend(f"- `{path.name}`" for path in files)
    if len(lines) == 1:
        lines.append("\nNo supported documents found yet.")
    return "\n".join(lines)


def upload_documents(files):
    if not files:
        return "Choose at least one file first.", list_indexed_documents(), uploaded_document_dropdown_update()

    if not isinstance(files, list):
        files = [files]

    saved_names = []
    skipped_names = []
    for file in files:
        source_path = Path(getattr(file, "name", file))
        if not source_path.exists() or source_path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            skipped_names.append(source_path.name if source_path.name else "unknown file")
            continue

        destination = make_unique_upload_path(source_path.name)
        shutil.copy2(source_path, destination)
        saved_names.append(destination.name)

    if saved_names:
        reset_rag_index_after_document_change()

    status_parts = []
    if saved_names:
        status_parts.append(f"Saved for all chats: {', '.join(f'`{name}`' for name in saved_names)}.")
        status_parts.append("The knowledge index will include them on the next answer.")
    if skipped_names:
        status_parts.append(f"Skipped unsupported/unreadable files: {', '.join(f'`{name}`' for name in skipped_names)}.")
    if not status_parts:
        status_parts.append("No files were saved.")

    selected_name = saved_names[0] if saved_names else None
    return " ".join(status_parts), list_indexed_documents(), uploaded_document_dropdown_update(selected_name)


def delete_uploaded_document(filename: str | None):
    if not filename:
        return "Choose an uploaded document to delete.", list_indexed_documents(), uploaded_document_dropdown_update()

    target = (UPLOADED_DOCUMENTS_DIR / filename).resolve()
    uploads_root = UPLOADED_DOCUMENTS_DIR.resolve()
    if uploads_root not in target.parents or not target.exists() or not target.is_file():
        return "Could not find that uploaded document.", list_indexed_documents(), uploaded_document_dropdown_update()

    target.unlink()
    reset_rag_index_after_document_change()

    return (
        f"Deleted `{filename}`. The knowledge index will be rebuilt without it on the next answer.",
        list_indexed_documents(),
        uploaded_document_dropdown_update(),
    )


def theme_style(mode: str) -> str:
    if mode == "dark":
        return f"""
        <style id="theme-mode-style">
          html, body, body > gradio-app, .gradio-container {{
            background: #06140f !important;
            color: #edf7ec !important;
          }}
          #landing-card, #auth-card, #profile-card, #chat-card, #app-header,
          .gradio-container .block,
          .gradio-container .form,
          .gradio-container .panel {{
            background: #10231b !important;
            border-color: rgba(134, 188, 37, 0.28) !important;
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.34) !important;
          }}
          #landing-brand, #chat-title, #header-logo {{
            color: {PRIMARY_GREEN} !important;
          }}
          #landing-subtitle, .status-line,
          .gradio-container label,
          .gradio-container p,
          .gradio-container h1,
          .gradio-container h2,
          .gradio-container h3,
          .gradio-container h4 {{
            color: #d8ead3 !important;
          }}
          #sidebar {{
            background: #04100c !important;
            border: 1px solid rgba(134, 188, 37, 0.24) !important;
          }}
          textarea, input, select,
          #sidebar textarea, #sidebar input, #sidebar select,
          #sidebar [role="textbox"], #sidebar [role="combobox"],
          .gradio-container [role="textbox"],
          .gradio-container [role="combobox"],
          .gradio-container .input,
          .gradio-container .input-container {{
            background: #071a13 !important;
            color: #f2ffe8 !important;
            border-color: rgba(134, 188, 37, 0.36) !important;
          }}
          [role="listbox"], .gradio-container [role="listbox"], #sidebar [role="listbox"] {{
            background: #ffffff !important;
            border-color: rgba(134, 188, 37, 0.38) !important;
          }}
          [role="listbox"] *, [role="option"], [role="option"] *,
          .gradio-container [role="listbox"] *, .gradio-container [role="option"],
          .gradio-container [role="option"] *, #sidebar [role="listbox"] *,
          #sidebar [role="option"], #sidebar [role="option"] * {{
            color: #10231b !important;
          }}
          [role="option"]:hover, [role="option"][aria-selected="true"],
          .gradio-container [role="option"]:hover,
          .gradio-container [role="option"][aria-selected="true"] {{
            background: #eef8df !important;
            color: #10231b !important;
          }}
          .secondary-action, .secondary-action button, button.secondary-action,
          #theme-bar button, #app-shell button.secondary-action,
          #app-shell .secondary-action button {{
            background: #193629 !important;
            border-color: rgba(134, 188, 37, 0.36) !important;
            color: #f2ffe8 !important;
          }}
          .danger-action, .danger-action button, button.danger-action {{
            background: #4b2020 !important;
            border-color: #8a4848 !important;
            color: #ffe5e5 !important;
          }}
        </style>
        """

    return f"""
    <style id="theme-mode-style">
      .gradio-container {{
        background: {LIGHT_BG} !important;
        color: #10231b !important;
      }}
      #landing-card, #auth-card, #profile-card, #chat-card, #app-header {{
        background: #ffffff !important;
        border-color: rgba(11, 61, 46, 0.12) !important;
      }}
      #landing-brand, #chat-title, #header-logo {{
        color: {DARK_GREEN} !important;
      }}
      #sidebar {{
        background: {DARK_GREEN} !important;
        border: 0 !important;
      }}
      [role="listbox"], .gradio-container [role="listbox"], #sidebar [role="listbox"] {{
        background: #ffffff !important;
        border-color: rgba(11, 61, 46, 0.18) !important;
      }}
      [role="listbox"] *, [role="option"], [role="option"] *,
      .gradio-container [role="listbox"] *, .gradio-container [role="option"],
      .gradio-container [role="option"] *, #sidebar [role="listbox"] *,
      #sidebar [role="option"], #sidebar [role="option"] * {{
        color: #10231b !important;
      }}
      [role="option"]:hover, [role="option"][aria-selected="true"],
      .gradio-container [role="option"]:hover,
      .gradio-container [role="option"][aria-selected="true"] {{
        background: #eef8df !important;
        color: #10231b !important;
      }}
      .secondary-action button, button.secondary-action {{
        background: #ffffff !important;
        border-color: rgba(11, 61, 46, 0.25) !important;
        color: {DARK_GREEN} !important;
      }}
    </style>
    """


def toggle_theme(current_mode: str | None):
    next_mode = "dark" if current_mode != "dark" else "light"
    button_label = "Light mode" if next_mode == "dark" else "Dark mode"
    return next_mode, theme_style(next_mode), gr.update(value=button_label)


def show_landing():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def show_login():
    return (
        gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False), gr.update(value="")
    )


def show_register():
    return (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
        gr.update(visible=False), gr.update(visible=False), gr.update(value=""),
        gr.update(visible=False), ""
    )


def show_chat():
    return (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=True), gr.update(visible=False), ""
    )


def show_profile(user_email: str | None):
    email = normalize_email(user_email)
    user = load_users().get(email)
    if not user:
        return (
            gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False), gr.update(value=""),
            gr.update(value=""), "Please login to view your profile."
        )

    return (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=True), gr.update(value=user.get("name") or ""),
        gr.update(value=user.get("email") or email), ""
    )


def register_user(name: str, email: str, password: str, confirm_password: str):
    email = normalize_email(email)
    users = load_users()
    if not email:
        return "Email cannot be empty.", gr.update(visible=False), ""
    if not password:
        return "Password cannot be empty.", gr.update(visible=False), ""
    if password != confirm_password:
        return "Passwords do not match.", gr.update(visible=False), ""
    if email in users:
        return "Account already exists.", gr.update(visible=not users[email].get("confirmed", False)), email

    salt = secrets.token_hex(16)
    users[email] = {
        "name": (name or "").strip(),
        "email": email,
        "salt": salt,
        "password_hash": hash_password(password, salt),
        "confirmed": False,
        "created_at": utc_now(),
    }
    save_users(users)
    return "Account created. Please confirm your email.", gr.update(visible=True), email


def confirm_email(email: str):
    email = normalize_email(email)
    users = load_users()
    if email not in users:
        return "Create an account before confirming email.", gr.update(visible=False)
    users[email]["confirmed"] = True
    users[email]["confirmed_at"] = utc_now()
    save_users(users)
    return "Email confirmed. You can login now.", gr.update(visible=False)


def login_user(email: str, password: str):
    email = normalize_email(email)
    user = load_users().get(email)
    hidden_chat = gr.update(visible=False)

    if not user or not verify_password(password, user):
        return (
            gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
            hidden_chat, gr.update(visible=False), None, None, [],
            conversation_dropdown_update(None), gr.update(value=""), "Invalid email or password."
        )

    if not user.get("confirmed", False):
        return (
            gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
            hidden_chat, gr.update(visible=False), None, None, [],
            conversation_dropdown_update(None), gr.update(value=""),
            "Please confirm your email before logging in."
        )

    conversations = get_user_conversations(email)
    active = conversations[0] if conversations else None
    active_id = active["id"] if active else None
    messages = active.get("messages", []) if active else []
    title = active.get("title", "") if active else ""

    return (
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=True), gr.update(visible=False), email, active_id, messages,
        conversation_dropdown_update(email, active_id), gr.update(value=title),
        f"Logged in as {user.get('name') or email}."
    )


def logout_user():
    return (
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False), gr.update(visible=False), None, None, [],
        conversation_dropdown_update(None), gr.update(value=""), ""
    )


def change_password(user_email: str | None, current_password: str, new_password: str, confirm_password: str):
    email = normalize_email(user_email)
    users = load_users()
    user = users.get(email)
    cleared = (gr.update(value=""), gr.update(value=""), gr.update(value=""))
    if not user:
        return ("Please login to change your password.", *cleared)
    if not verify_password(current_password or "", user):
        return ("Current password is incorrect.", *cleared)
    if not new_password:
        return ("New password cannot be empty.", *cleared)
    if new_password != confirm_password:
        return ("New passwords do not match.", *cleared)

    salt = secrets.token_hex(16)
    users[email]["salt"] = salt
    users[email]["password_hash"] = hash_password(new_password, salt)
    users[email]["password_updated_at"] = utc_now()
    save_users(users)
    return ("Password changed successfully.", *cleared)


def start_new_chat(user_email: str | None):
    if not user_email:
        return [], None, conversation_dropdown_update(None), gr.update(value=""), "Login required."
    conversation = create_conversation(user_email)
    return [], conversation["id"], conversation_dropdown_update(user_email, conversation["id"]), gr.update(value=conversation["title"]), "New chat created."


def load_conversation_messages(conversation_id: str | None, user_email: str | None):
    conversation = get_conversation(conversation_id, user_email)
    if not conversation:
        return [], None, gr.update(value=""), "No conversation selected."
    return conversation.get("messages", []), conversation["id"], gr.update(value=conversation.get("title", "")), f"Loaded: {conversation.get('title', 'New chat')}"


def rename_conversation(conversation_id: str | None, user_email: str | None, new_title: str):
    conversation = get_conversation(conversation_id, user_email)
    if not conversation:
        return conversation_dropdown_update(user_email), conversation_id, gr.update(value=new_title or ""), "Select a conversation to rename."

    title = (new_title or "").strip()
    if not title:
        return conversation_dropdown_update(user_email, conversation_id), conversation_id, gr.update(value=conversation.get("title", "")), "Chat name cannot be empty."

    conversation["title"] = title
    conversation["updated_at"] = utc_now()
    update_conversation(conversation)
    return conversation_dropdown_update(user_email, conversation_id), conversation_id, gr.update(value=title), "Chat renamed."


def delete_conversation(conversation_id: str | None, user_email: str | None):
    email = normalize_email(user_email)
    conversations = [
        item for item in load_conversations()
        if not (item.get("id") == conversation_id and normalize_email(item.get("user_email")) == email)
    ]
    save_conversations(conversations)
    remaining = get_user_conversations(email)
    next_conversation = remaining[0] if remaining else None
    next_id = next_conversation["id"] if next_conversation else None
    messages = next_conversation.get("messages", []) if next_conversation else []
    title = next_conversation.get("title", "") if next_conversation else ""
    return messages, next_id, conversation_dropdown_update(email, next_id), gr.update(value=title), "Chat deleted." if conversation_id else "No chat selected."


def send_message(message: str, chat_history: list[Any] | None, user_email: str | None, active_id: str | None):
    text = (message or "").strip()
    if not user_email:
        return chat_history or [], gr.update(value=""), active_id, conversation_dropdown_update(None), gr.update(), "Login required."
    if not text:
        return chat_history or [], gr.update(value=""), active_id, conversation_dropdown_update(user_email, active_id), gr.update(), ""

    conversation = get_conversation(active_id, user_email)
    if not conversation:
        conversation = create_conversation(user_email, title=title_from_message(text))
        active_id = conversation["id"]

    current_messages = conversation.get("messages", [])
    rag_failed = False
    try:
        response = rag_answer(text, current_messages)
    except Exception as error:
        rag_failed = True
        response = f"I could not complete the RAG answer. Technical detail: {error}"

    updated_messages = [
        *current_messages,
        {"role": "user", "content": text},
        {"role": "assistant", "content": response},
    ]
    if not conversation.get("title") or conversation.get("title") == "New chat":
        conversation["title"] = title_from_message(text)
    conversation["messages"] = updated_messages
    conversation["updated_at"] = utc_now()
    update_conversation(conversation)

    return (
        updated_messages, gr.update(value=""), conversation["id"],
        conversation_dropdown_update(user_email, conversation["id"]),
        gr.update(value=conversation["title"]),
        "RAG error. Check Azure OpenAI credentials and index." if rag_failed else "Answer generated with RAG."
    )


def clear_conversation(user_email: str | None, active_id: str | None):
    if not user_email:
        return [], active_id, conversation_dropdown_update(None), gr.update(), "Login required."
    conversation = get_conversation(active_id, user_email)
    if not conversation:
        return [], active_id, conversation_dropdown_update(user_email, active_id), gr.update(), "No active chat to clear."
    conversation["messages"] = []
    conversation["updated_at"] = utc_now()
    update_conversation(conversation)
    return [], conversation["id"], conversation_dropdown_update(user_email, conversation["id"]), gr.update(value=conversation.get("title", "New chat")), "Conversation cleared."


def export_conversation(user_email: str | None, active_id: str | None):
    if not user_email:
        return "Login required.", gr.update(value=None, visible=False)
    conversation = get_conversation(active_id, user_email)
    if not conversation:
        return "No active chat to export.", gr.update(value=None, visible=False)
    messages = conversation.get("messages", [])
    if not messages:
        return "This chat is empty.", gr.update(value=None, visible=False)

    EXPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(char if char.isalnum() else "_" for char in conversation.get("title", "chat")).strip("_")
    safe_title = safe_title[:40] or "chat"
    export_path = EXPORTS_DIR / f"{safe_title}_{timestamp}.md"
    lines = [
        f"# {conversation.get('title', 'EcoGuide conversation')}",
        "",
        f"- User: {normalize_email(user_email)}",
        f"- Created at: {conversation.get('created_at', '')}",
        f"- Exported at: {utc_now()}",
        "",
    ]
    for message in messages:
        role = "User" if message.get("role") == "user" else "EcoGuide"
        lines.extend([f"## {role}", "", str(message.get("content", "")), ""])
    export_path.write_text("\n".join(lines), encoding="utf-8")
    return f"Conversation exported: `{export_path.name}`", gr.update(value=str(export_path), visible=True)


def update_rag_settings(num_chunks: int, show_sources: bool, temperature: float):
    APP_SETTINGS["num_chunks"] = int(num_chunks)
    APP_SETTINGS["show_sources"] = bool(show_sources)
    APP_SETTINGS["temperature"] = float(temperature)
    return f"Settings saved: {APP_SETTINGS['num_chunks']} chunks, sources {'on' if APP_SETTINGS['show_sources'] else 'off'}, temperature {APP_SETTINGS['temperature']:.2f}."


def create_interface() -> gr.Blocks:
    ensure_demo_user()

    with gr.Blocks(css=CUSTOM_CSS, title="EcoGuide") as app:
        current_user = gr.State(None)
        active_conversation_id = gr.State(None)
        pending_confirmation_email = gr.State("")
        theme_mode = gr.State("light")

        with gr.Column(elem_id="app-shell"):
            theme_override = gr.HTML(theme_style("light"))
            with gr.Row(elem_id="theme-bar"):
                theme_btn = gr.Button("Dark mode", elem_classes=["secondary-action"])

            with gr.Column(visible=True, elem_id="landing-card") as landing_page:
                gr.HTML('<div id="landing-brand">Eco<span>Guide</span></div><div id="landing-subtitle">Your climate change and sustainability assistant</div>')
                with gr.Row():
                    landing_login_btn = gr.Button("Login", elem_classes=["primary-action"])
                    landing_register_btn = gr.Button("Register", elem_classes=["secondary-action"])

            with gr.Column(visible=False, elem_id="auth-card") as login_page:
                gr.Markdown("## Login")
                gr.Markdown(f"Demo account: `{DEMO_EMAIL}` / `{DEMO_PASSWORD}`")
                login_email = gr.Textbox(label="Email")
                login_password = gr.Textbox(label="Password", type="password")
                login_status = gr.Markdown("", elem_classes=["status-line"])
                with gr.Row():
                    login_btn = gr.Button("Login", elem_classes=["primary-action"])
                    login_back_btn = gr.Button("Back", elem_classes=["secondary-action"])

            with gr.Column(visible=False, elem_id="auth-card") as register_page:
                gr.Markdown("## Register")
                register_name = gr.Textbox(label="Name")
                register_email = gr.Textbox(label="Email")
                register_password = gr.Textbox(label="Password", type="password")
                register_confirm_password = gr.Textbox(label="Confirm password", type="password")
                register_status = gr.Markdown("", elem_classes=["status-line"])
                with gr.Row():
                    create_account_btn = gr.Button("Create account", elem_classes=["primary-action"])
                    register_back_btn = gr.Button("Back", elem_classes=["secondary-action"])
                confirm_email_btn = gr.Button("Confirm email", visible=False, elem_classes=["primary-action"])

            with gr.Column(visible=False, elem_id="profile-card") as profile_page:
                gr.Markdown("## Profile")
                profile_name = gr.Textbox(label="Name", interactive=False)
                profile_email = gr.Textbox(label="Email", interactive=False)
                gr.Markdown("### Change password")
                current_password = gr.Textbox(label="Current password", type="password")
                new_password = gr.Textbox(label="New password", type="password")
                confirm_new_password = gr.Textbox(label="Confirm new password", type="password")
                profile_status = gr.Markdown("", elem_classes=["status-line"])
                with gr.Row():
                    save_password_btn = gr.Button("Change password", elem_classes=["primary-action"])
                    profile_back_btn = gr.Button("Back to chat", elem_classes=["secondary-action"])
                    profile_logout_btn = gr.Button("Logout", elem_classes=["secondary-action"])

            with gr.Column(visible=False) as chat_page:
                with gr.Row(elem_id="app-header"):
                    with gr.Column(scale=1, min_width=220):
                        gr.HTML('<div id="header-logo">Eco<span>Guide</span></div>')
                    with gr.Column(scale=1):
                        with gr.Row(elem_id="header-actions"):
                            header_profile_btn = gr.Button("Profile", elem_classes=["secondary-action"])
                            header_logout_btn = gr.Button("Logout", elem_classes=["secondary-action"])

                with gr.Row():
                    with gr.Column(scale=1, min_width=260, elem_id="sidebar"):
                        gr.Markdown("### EcoGuide")
                        chat_status = gr.Markdown("", elem_classes=["status-line"])
                        conversation_select = gr.Dropdown(label="Conversations", choices=[], value=None, interactive=True)
                        rename_input = gr.Textbox(label="Chat name", placeholder="New name")
                        new_chat_btn = gr.Button("New chat", elem_classes=["primary-action"])
                        rename_btn = gr.Button("Rename chat", elem_classes=["secondary-action"])
                        delete_btn = gr.Button("Delete chat", elem_classes=["danger-action"])

                    with gr.Column(scale=4, elem_id="chat-card"):
                        gr.HTML('<div id="chat-title">EcoGuide Chatbot</div>')
                        gr.Markdown("Ask questions about climate change and get AI-powered answers with sources.")

                        with gr.Tab("Chat"):
                            chatbot = gr.Chatbot(label="EcoGuide Chatbot", type="messages", height=500, show_label=False)
                            message_box = gr.Textbox(label="Message", placeholder="Ask EcoGuide about climate change or sustainable recommendations", lines=1)
                            with gr.Row():
                                send_btn = gr.Button("Submit", elem_classes=["primary-action"])
                                clear_chat_btn = gr.Button("Clear chat", elem_classes=["secondary-action"])
                                export_chat_btn = gr.Button("Export chat", elem_classes=["secondary-action"])
                                main_new_chat_btn = gr.Button("New conversation", elem_classes=["secondary-action"])
                            export_status = gr.Markdown("", elem_classes=["status-line"])
                            export_file = gr.File(label="Download export", visible=False, interactive=False)
                            gr.Examples(
                                examples=[
                                    "What is climate change?",
                                    "How can I reduce my carbon footprint?",
                                    "What are carbon offset strategies?",
                                    "Recommend sustainable products for my home",
                                ],
                                inputs=message_box,
                            )

                        with gr.Tab("Documents"):
                            gr.Markdown("### Upload New Documents")
                            file_upload = gr.File(
                                label="Upload PDF, HTML, DOCX, PPTX or CSV files",
                                file_types=[".pdf", ".html", ".docx", ".pptx", ".csv"],
                                file_count="multiple",
                            )
                            upload_btn = gr.Button("Upload documents", elem_classes=["primary-action"])
                            upload_status = gr.Markdown("", elem_classes=["status-line"])
                            gr.Markdown("### Uploaded Documents")
                            uploaded_document_select = gr.Dropdown(
                                label="Saved for all chats",
                                choices=uploaded_document_choices(),
                                value=uploaded_document_choices()[0][1] if uploaded_document_choices() else None,
                                interactive=True,
                            )
                            delete_uploaded_document_btn = gr.Button("Delete selected document", elem_classes=["danger-action"])
                            documents_list = gr.Markdown(list_indexed_documents())

                        with gr.Tab("Settings"):
                            gr.Markdown("### RAG Settings")
                            num_chunks = gr.Slider(minimum=1, maximum=10, value=APP_SETTINGS["num_chunks"], step=1, label="Number of chunks to retrieve")
                            show_sources = gr.Checkbox(label="Show source citations in answers", value=APP_SETTINGS["show_sources"])
                            temperature = gr.Slider(minimum=0.0, maximum=1.5, value=APP_SETTINGS["temperature"], step=0.1, label="LLM temperature")
                            save_settings_btn = gr.Button("Save settings", elem_classes=["primary-action"])
                            settings_status = gr.Markdown("", elem_classes=["status-line"])

        panel_outputs = [landing_page, login_page, register_page, chat_page, profile_page]

        theme_btn.click(toggle_theme, inputs=[theme_mode], outputs=[theme_mode, theme_override, theme_btn])
        landing_login_btn.click(show_login, inputs=None, outputs=[*panel_outputs, login_status])
        landing_register_btn.click(show_register, inputs=None, outputs=[*panel_outputs, register_status, confirm_email_btn, pending_confirmation_email])
        login_back_btn.click(show_landing, inputs=None, outputs=panel_outputs)
        register_back_btn.click(show_landing, inputs=None, outputs=panel_outputs)

        create_account_btn.click(register_user, inputs=[register_name, register_email, register_password, register_confirm_password], outputs=[register_status, confirm_email_btn, pending_confirmation_email])
        confirm_email_btn.click(confirm_email, inputs=[pending_confirmation_email], outputs=[register_status, confirm_email_btn])

        login_outputs = [*panel_outputs, current_user, active_conversation_id, chatbot, conversation_select, rename_input, chat_status]
        login_btn.click(login_user, inputs=[login_email, login_password], outputs=login_outputs)

        logout_outputs = [*panel_outputs, current_user, active_conversation_id, chatbot, conversation_select, rename_input, chat_status]
        header_logout_btn.click(logout_user, inputs=None, outputs=logout_outputs)
        profile_logout_btn.click(logout_user, inputs=None, outputs=logout_outputs)

        profile_outputs = [*panel_outputs, profile_name, profile_email, profile_status]
        header_profile_btn.click(show_profile, inputs=[current_user], outputs=profile_outputs)
        profile_back_btn.click(show_chat, inputs=None, outputs=[*panel_outputs, profile_status])
        save_password_btn.click(change_password, inputs=[current_user, current_password, new_password, confirm_new_password], outputs=[profile_status, current_password, new_password, confirm_new_password])

        new_chat_outputs = [chatbot, active_conversation_id, conversation_select, rename_input, chat_status]
        new_chat_btn.click(start_new_chat, inputs=[current_user], outputs=new_chat_outputs)
        main_new_chat_btn.click(start_new_chat, inputs=[current_user], outputs=new_chat_outputs)

        conversation_select.change(load_conversation_messages, inputs=[conversation_select, current_user], outputs=[chatbot, active_conversation_id, rename_input, chat_status])
        rename_btn.click(rename_conversation, inputs=[active_conversation_id, current_user, rename_input], outputs=[conversation_select, active_conversation_id, rename_input, chat_status])
        delete_btn.click(delete_conversation, inputs=[active_conversation_id, current_user], outputs=[chatbot, active_conversation_id, conversation_select, rename_input, chat_status])

        send_outputs = [chatbot, message_box, active_conversation_id, conversation_select, rename_input, chat_status]
        send_btn.click(send_message, inputs=[message_box, chatbot, current_user, active_conversation_id], outputs=send_outputs)
        message_box.submit(send_message, inputs=[message_box, chatbot, current_user, active_conversation_id], outputs=send_outputs)
        clear_chat_btn.click(clear_conversation, inputs=[current_user, active_conversation_id], outputs=[chatbot, active_conversation_id, conversation_select, rename_input, chat_status])
        export_chat_btn.click(export_conversation, inputs=[current_user, active_conversation_id], outputs=[export_status, export_file])
        upload_btn.click(
            upload_documents,
            inputs=[file_upload],
            outputs=[upload_status, documents_list, uploaded_document_select],
        )
        delete_uploaded_document_btn.click(
            delete_uploaded_document,
            inputs=[uploaded_document_select],
            outputs=[upload_status, documents_list, uploaded_document_select],
        )
        save_settings_btn.click(update_rag_settings, inputs=[num_chunks, show_sources, temperature], outputs=[settings_status])

    return app


def main() -> None:
    app = create_interface()
    app.queue()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
