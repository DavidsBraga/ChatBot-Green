"""
Gradio Web Interface for EcoGuide Chatbot.

This file replaces the terminal interaction in main.py with a simple Gradio UI:
- landing page
- local JSON registration/login with simulated email confirmation
- per-user conversation history
- real RAG answers through LLM, Embeddings, FAISSIndex and document ingestion

Remaining TODOs from the starter kit that are still useful after this frontend:
- add richer source metadata in FAISS for page-level citations
- implement product cards from sustainable_products.csv
- add real multimodal RAG for images/charts in documents
- add upload/re-index workflow for new documents
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# Compatibility shim for Gradio 5.5.0 with newer huggingface_hub versions where
# HfFolder was removed. This keeps the demo self-contained without changing deps.
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
load_dotenv(APP_DIR / ".env", override=True)

USERS_FILE = APP_DIR / "users.json"
CONVERSATIONS_FILE = APP_DIR / "conversations.json"

PRIMARY_GREEN = "#86BC25"
DARK_GREEN = "#0B3D2E"
LIGHT_BG = "#F6FFF2"

_llm = None
_index = None
_rag_ready = False
APP_SETTINGS = {
    "num_chunks": 5,
    "show_sources": True,
    "chunking_strategy": "token",
    "temperature": 0.7,
}


CUSTOM_CSS = f"""
:root {{
  --eco-green: {PRIMARY_GREEN};
  --eco-dark: {DARK_GREEN};
  --eco-bg: {LIGHT_BG};
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

#landing-card, #auth-card {{
  background: #ffffff;
  border: 1px solid rgba(11, 61, 46, 0.12);
  border-radius: 8px;
  box-shadow: 0 18px 48px rgba(11, 61, 46, 0.12);
  padding: 34px;
}}

#landing-brand {{
  color: var(--eco-dark);
  font-size: 56px;
  line-height: 1;
  font-weight: 800;
  letter-spacing: 0;
  margin-bottom: 10px;
}}

#landing-subtitle {{
  color: #2f4f43;
  font-size: 20px;
  margin-bottom: 28px;
}}

#sidebar {{
  background: var(--eco-dark);
  border-radius: 8px;
  padding: 18px;
  min-height: 650px;
}}

#sidebar * {{
  color: #ffffff;
}}

#sidebar input, #sidebar textarea, #sidebar select {{
  color: #10231b !important;
}}

#chat-card {{
  background: #ffffff;
  border: 1px solid rgba(11, 61, 46, 0.10);
  border-radius: 8px;
  padding: 18px;
  min-height: 650px;
  box-shadow: 0 14px 38px rgba(11, 61, 46, 0.10);
}}

#chat-title {{
  color: var(--eco-dark);
  font-size: 30px;
  line-height: 1.2;
  font-weight: 760;
  letter-spacing: 0;
  margin: 0 0 8px 0;
}}

#theme-bar {{
  display: flex;
  justify-content: flex-end;
  align-items: center;
  padding: 14px 0 10px 0;
}}

#theme-status {{
  color: #315246;
  font-size: 13px;
  margin-right: 10px;
}}

.primary-action button, button.primary-action {{
  background: var(--eco-green) !important;
  border-color: var(--eco-green) !important;
  color: #0b1f16 !important;
  font-weight: 700 !important;
}}

.secondary-action button, button.secondary-action {{
  background: #ffffff !important;
  border-color: rgba(11, 61, 46, 0.25) !important;
  color: var(--eco-dark) !important;
  font-weight: 650 !important;
}}

.danger-action button, button.danger-action {{
  background: #f7d9d9 !important;
  border-color: #e7aaaa !important;
  color: #6d1515 !important;
  font-weight: 650 !important;
}}

.status-line {{
  color: #24483b;
  font-size: 14px;
}}

#sidebar .status-line {{
  color: #dbeed4;
}}

textarea, input {{
  border-radius: 8px !important;
}}

button {{
  border-radius: 8px !important;
}}

@media (max-width: 820px) {{
  #landing-brand {{
    font-size: 42px;
  }}
  #sidebar {{
    min-height: auto;
  }}
  #chat-card {{
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
    salt = user.get("salt", "")
    return hash_password(password, salt) == user.get("password_hash")


def title_from_message(message: str) -> str:
    cleaned = " ".join((message or "").strip().split())
    if not cleaned:
        return "New chat"
    return cleaned[:42] + ("..." if len(cleaned) > 42 else "")


def get_user_conversations(user_email: str | None) -> list[dict[str, Any]]:
    email = normalize_email(user_email)
    conversations = [
        conversation
        for conversation in load_conversations()
        if normalize_email(conversation.get("user_email")) == email
    ]
    return sorted(conversations, key=lambda item: item.get("updated_at", ""), reverse=True)


def get_conversation(conversation_id: str | None, user_email: str | None) -> dict[str, Any] | None:
    email = normalize_email(user_email)
    for conversation in load_conversations():
        if conversation.get("id") == conversation_id and normalize_email(conversation.get("user_email")) == email:
            return conversation
    return None


def conversation_choices(user_email: str | None) -> list[tuple[str, str]]:
    choices = []
    for conversation in get_user_conversations(user_email):
        title = conversation.get("title") or "New chat"
        choices.append((title, conversation["id"]))
    return choices


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


def rename_conversation(conversation_id: str | None, user_email: str | None, new_title: str):
    conversation = get_conversation(conversation_id, user_email)
    if not conversation:
        return (
            conversation_dropdown_update(user_email),
            conversation_id,
            gr.update(value=new_title or ""),
            "Select a conversation to rename.",
        )

    title = (new_title or "").strip()
    if not title:
        return (
            conversation_dropdown_update(user_email, conversation_id),
            conversation_id,
            gr.update(value=conversation.get("title", "")),
            "Chat name cannot be empty.",
        )

    conversation["title"] = title
    conversation["updated_at"] = utc_now()
    update_conversation(conversation)
    return (
        conversation_dropdown_update(user_email, conversation_id),
        conversation_id,
        gr.update(value=title),
        "Chat renamed.",
    )


def delete_conversation(conversation_id: str | None, user_email: str | None):
    email = normalize_email(user_email)
    conversations = load_conversations()
    conversations = [
        item
        for item in conversations
        if not (item.get("id") == conversation_id and normalize_email(item.get("user_email")) == email)
    ]
    save_conversations(conversations)

    remaining = get_user_conversations(email)
    next_conversation = remaining[0] if remaining else None
    next_id = next_conversation["id"] if next_conversation else None
    messages = next_conversation.get("messages", []) if next_conversation else []
    title = next_conversation.get("title", "") if next_conversation else ""

    return (
        messages,
        next_id,
        conversation_dropdown_update(email, next_id),
        gr.update(value=title),
        "Chat deleted." if conversation_id else "No chat selected.",
    )


def load_conversation_messages(conversation_id: str | None, user_email: str | None):
    conversation = get_conversation(conversation_id, user_email)
    if not conversation:
        return [], None, gr.update(value=""), "No conversation selected."
    return (
        conversation.get("messages", []),
        conversation["id"],
        gr.update(value=conversation.get("title", "")),
        f"Loaded: {conversation.get('title', 'New chat')}",
    )


def initialize_rag_once() -> None:
    global _llm, _index, _rag_ready

    if _rag_ready:
        return

    from src.ingestion.ingest_files import ingest_files_data_folder
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
        _index.save_index()

    _rag_ready = True


def history_to_llm_messages(chat_history: list[Any] | None) -> list[dict[str, str]]:
    llm_history: list[dict[str, str]] = []
    for item in chat_history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content:
                llm_history.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user_msg, bot_msg = item[0], item[1]
            if user_msg:
                llm_history.append({"role": "user", "content": str(user_msg)})
            if bot_msg:
                llm_history.append({"role": "assistant", "content": str(bot_msg)})
    return llm_history


def rag_answer(message: str, chat_history: list[Any] | None) -> str:
    initialize_rag_once()
    llm_history = history_to_llm_messages(chat_history)
    num_chunks = int(APP_SETTINGS.get("num_chunks", 5))
    retrieved_chunks = _index.retrieve_chunks(message, num_chunks=num_chunks)
    context = "\n\n#####\n\n".join(retrieved_chunks)
    temperature = float(APP_SETTINGS.get("temperature", 0.7))
    response = _llm.get_response(llm_history, context, message, temperature=temperature)

    if APP_SETTINGS.get("show_sources", True):
        response += (
            "\n\n---\n"
            "**Sources:** Retrieved from the indexed FAISS knowledge base. "
            "Document/page-level citations still require metadata support in the vector index."
        )

    return response


def list_indexed_documents() -> str:
    data_dir = APP_DIR / "data"
    uploaded_dir = APP_DIR / "uploaded_documents"
    supported = {".pdf", ".html", ".htm", ".docx", ".pptx", ".csv"}
    lines = ["### Current documents"]

    for folder, label in ((data_dir, "Data folder"), (uploaded_dir, "Uploaded")):
        if not folder.exists():
            continue
        files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in supported)
        if not files:
            continue
        lines.append(f"\n**{label}**")
        for path in files:
            lines.append(f"- `{path.name}`")

    if len(lines) == 1:
        lines.append("\nNo supported documents found yet.")

    return "\n".join(lines)


def upload_document(file):
    if file is None:
        return "Choose a file first.", list_indexed_documents()

    source_path = Path(getattr(file, "name", file))
    if not source_path.exists():
        return "Could not read the uploaded file.", list_indexed_documents()

    supported = {".pdf", ".html", ".htm", ".docx", ".pptx", ".csv"}
    if source_path.suffix.lower() not in supported:
        return "Unsupported file type. Use PDF, HTML, DOCX, PPTX or CSV.", list_indexed_documents()

    upload_dir = APP_DIR / "uploaded_documents"
    upload_dir.mkdir(exist_ok=True)
    destination = upload_dir / source_path.name
    shutil.copy2(source_path, destination)

    return (
        f"Saved `{source_path.name}` to `uploaded_documents/`. "
        "The base RAG ingestion still uses the original `data/` folder unless re-indexing is wired later.",
        list_indexed_documents(),
    )


def update_rag_settings(
    num_chunks: int,
    show_sources: bool,
    chunking_strategy: str,
    temperature: float,
):
    APP_SETTINGS["num_chunks"] = int(num_chunks)
    APP_SETTINGS["show_sources"] = bool(show_sources)
    APP_SETTINGS["chunking_strategy"] = chunking_strategy
    APP_SETTINGS["temperature"] = float(temperature)

    return (
        f"Settings saved: {APP_SETTINGS['num_chunks']} chunks, "
        f"sources {'on' if APP_SETTINGS['show_sources'] else 'off'}, "
        f"chunking strategy `{APP_SETTINGS['chunking_strategy']}`, "
        f"temperature {APP_SETTINGS['temperature']:.2f}."
    )


def theme_style(mode: str) -> str:
    if mode == "dark":
        return f"""
        <style id="theme-mode-style">
          .gradio-container {{
            background: #06140f !important;
            color: #edf7ec !important;
          }}
          #landing-card, #auth-card, #chat-card {{
            background: #10231b !important;
            border-color: rgba(134, 188, 37, 0.28) !important;
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.34) !important;
          }}
          #landing-brand, #chat-title {{
            color: {PRIMARY_GREEN} !important;
          }}
          #landing-subtitle, .status-line, #theme-status {{
            color: #d8ead3 !important;
          }}
          #sidebar {{
            background: #04100c !important;
            border: 1px solid rgba(134, 188, 37, 0.24) !important;
          }}
          textarea, input, select {{
            background: #f8fff2 !important;
            color: #10231b !important;
          }}
          label, .prose, .markdown, .wrap, .contain {{
            color: #edf7ec !important;
          }}
          .secondary-action button, button.secondary-action {{
            background: #193629 !important;
            border-color: rgba(134, 188, 37, 0.36) !important;
            color: #f2ffe8 !important;
          }}
          .danger-action button, button.danger-action {{
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
      #landing-card, #auth-card, #chat-card {{
        background: #ffffff !important;
        border-color: rgba(11, 61, 46, 0.12) !important;
        box-shadow: 0 18px 48px rgba(11, 61, 46, 0.12) !important;
      }}
      #landing-brand, #chat-title {{
        color: {DARK_GREEN} !important;
      }}
      #landing-subtitle, .status-line, #theme-status {{
        color: #315246 !important;
      }}
      #sidebar {{
        background: {DARK_GREEN} !important;
        border: 0 !important;
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
    button_label = "Modo claro" if next_mode == "dark" else "Modo escuro"
    status = "Tema escuro ativo." if next_mode == "dark" else "Tema claro ativo."
    return next_mode, theme_style(next_mode), gr.update(value=button_label), status


def show_landing():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def show_login():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=""),
    )


def show_register():
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(visible=False),
        "",
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
    users = load_users()
    user = users.get(email)

    if not user or not verify_password(password, user):
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            None,
            [],
            conversation_dropdown_update(None),
            gr.update(value=""),
            "Invalid email or password.",
        )

    if not user.get("confirmed", False):
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            None,
            [],
            conversation_dropdown_update(None),
            gr.update(value=""),
            "Please confirm your email before logging in.",
        )

    conversations = get_user_conversations(email)
    active = conversations[0] if conversations else None
    active_id = active["id"] if active else None
    messages = active.get("messages", []) if active else []
    title = active.get("title", "") if active else ""

    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        email,
        active_id,
        messages,
        conversation_dropdown_update(email, active_id),
        gr.update(value=title),
        f"Logged in as {user.get('name') or email}.",
    )


def logout_user():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        None,
        None,
        [],
        conversation_dropdown_update(None),
        gr.update(value=""),
        "",
    )


def start_new_chat(user_email: str | None):
    if not user_email:
        return [], None, conversation_dropdown_update(None), gr.update(value=""), "Login required."
    conversation = create_conversation(user_email)
    return (
        [],
        conversation["id"],
        conversation_dropdown_update(user_email, conversation["id"]),
        gr.update(value=conversation["title"]),
        "New chat created.",
    )


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
        response = (
            "I could not complete the RAG answer. "
            f"Technical detail: {error}"
        )

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
        updated_messages,
        gr.update(value=""),
        conversation["id"],
        conversation_dropdown_update(user_email, conversation["id"]),
        gr.update(value=conversation["title"]),
        "RAG error. Check Azure OpenAI credentials and index." if rag_failed else "Answer generated with RAG.",
    )


def create_interface() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="EcoGuide") as app:
        current_user = gr.State(None)
        active_conversation_id = gr.State(None)
        pending_confirmation_email = gr.State("")
        theme_mode = gr.State("light")

        with gr.Column(elem_id="app-shell"):
            theme_override = gr.HTML(theme_style("light"))
            with gr.Row(elem_id="theme-bar"):
                theme_status = gr.Markdown("Tema claro ativo.", elem_id="theme-status")
                theme_btn = gr.Button("Modo escuro", elem_classes=["secondary-action"])

            with gr.Column(visible=True, elem_id="landing-card") as landing_page:
                gr.HTML(
                    """
                    <div id="landing-brand">EcoGuide</div>
                    <div id="landing-subtitle">Your climate change and sustainability assistant</div>
                    """
                )
                with gr.Row():
                    landing_login_btn = gr.Button("Login", elem_classes=["primary-action"])
                    landing_register_btn = gr.Button("Register", elem_classes=["secondary-action"])

            with gr.Column(visible=False, elem_id="auth-card") as login_page:
                gr.Markdown("## Login")
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

            with gr.Row(visible=False) as chat_page:
                with gr.Column(scale=1, min_width=260, elem_id="sidebar"):
                    gr.Markdown("### EcoGuide")
                    chat_status = gr.Markdown("", elem_classes=["status-line"])
                    conversation_select = gr.Dropdown(
                        label="Conversations",
                        choices=[],
                        value=None,
                        interactive=True,
                    )
                    rename_input = gr.Textbox(label="Chat name", placeholder="New name")
                    new_chat_btn = gr.Button("New chat", elem_classes=["primary-action"])
                    rename_btn = gr.Button("Rename chat", elem_classes=["secondary-action"])
                    delete_btn = gr.Button("Delete chat", elem_classes=["danger-action"])
                    logout_btn = gr.Button("Logout", elem_classes=["secondary-action"])

                with gr.Column(scale=4, elem_id="chat-card"):
                    gr.HTML('<div id="chat-title">EcoGuide Chatbot</div>')
                    gr.Markdown("Ask questions about climate change and get AI-powered answers with sources.")

                    with gr.Tab("Chat"):
                        chatbot = gr.Chatbot(
                            label="EcoGuide Chatbot",
                            type="messages",
                            height=500,
                            show_label=False,
                        )
                        message_box = gr.Textbox(
                            label="Message",
                            placeholder="Ask EcoGuide about climate change or sustainable recommendations",
                            lines=3,
                        )
                        with gr.Row():
                            send_btn = gr.Button("Send", elem_classes=["primary-action"])
                            main_new_chat_btn = gr.Button("New conversation", elem_classes=["secondary-action"])
                            main_logout_btn = gr.Button("Logout", elem_classes=["secondary-action"])

                        gr.Examples(
                            examples=[
                                "What is climate change?",
                                "How can I reduce my carbon footprint?",
                                "What are carbon offset strategies?",
                                "Recommend sustainable products",
                            ],
                            inputs=message_box,
                        )

                    with gr.Tab("Documents"):
                        gr.Markdown("### Upload New Documents")
                        file_upload = gr.File(
                            label="Upload PDF, HTML, DOCX, PPTX or CSV files",
                            file_types=[".pdf", ".html", ".docx", ".pptx", ".csv"],
                        )
                        upload_btn = gr.Button("Upload document", elem_classes=["primary-action"])
                        upload_status = gr.Markdown("", elem_classes=["status-line"])
                        documents_list = gr.Markdown(list_indexed_documents())

                    with gr.Tab("Settings"):
                        gr.Markdown("### RAG Settings")
                        num_chunks = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=APP_SETTINGS["num_chunks"],
                            step=1,
                            label="Number of chunks to retrieve",
                            info="More chunks means more context, but slower answers.",
                        )
                        show_sources = gr.Checkbox(
                            label="Show source citations in answers",
                            value=APP_SETTINGS["show_sources"],
                            info="Display that answers are grounded in the FAISS knowledge base.",
                        )
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.5,
                            value=APP_SETTINGS["temperature"],
                            step=0.1,
                            label="LLM temperature",
                            info="Lower values are more deterministic; higher values are more creative.",
                        )
                        chunking_strategy = gr.Dropdown(
                            choices=["token", "sentence", "semantic"],
                            value=APP_SETTINGS["chunking_strategy"],
                            label="Chunking strategy",
                            info="Currently saved for demo control; re-indexing still uses the starter-kit ingestion flow.",
                        )
                        save_settings_btn = gr.Button("Save settings", elem_classes=["primary-action"])
                        settings_status = gr.Markdown("", elem_classes=["status-line"])

        panel_outputs = [landing_page, login_page, register_page, chat_page]

        theme_btn.click(
            toggle_theme,
            inputs=[theme_mode],
            outputs=[theme_mode, theme_override, theme_btn, theme_status],
        )

        landing_login_btn.click(
            show_login,
            inputs=None,
            outputs=[*panel_outputs, login_status],
        )
        landing_register_btn.click(
            show_register,
            inputs=None,
            outputs=[*panel_outputs, register_status, confirm_email_btn, pending_confirmation_email],
        )
        login_back_btn.click(show_landing, inputs=None, outputs=panel_outputs)
        register_back_btn.click(show_landing, inputs=None, outputs=panel_outputs)

        create_account_btn.click(
            register_user,
            inputs=[register_name, register_email, register_password, register_confirm_password],
            outputs=[register_status, confirm_email_btn, pending_confirmation_email],
        )
        confirm_email_btn.click(
            confirm_email,
            inputs=[pending_confirmation_email],
            outputs=[register_status, confirm_email_btn],
        )

        login_outputs = [
            landing_page,
            login_page,
            register_page,
            chat_page,
            current_user,
            active_conversation_id,
            chatbot,
            conversation_select,
            rename_input,
            chat_status,
        ]
        login_btn.click(login_user, inputs=[login_email, login_password], outputs=login_outputs)

        logout_outputs = [
            landing_page,
            login_page,
            register_page,
            chat_page,
            current_user,
            active_conversation_id,
            chatbot,
            conversation_select,
            rename_input,
            chat_status,
        ]
        logout_btn.click(logout_user, inputs=None, outputs=logout_outputs)
        main_logout_btn.click(logout_user, inputs=None, outputs=logout_outputs)

        new_chat_outputs = [chatbot, active_conversation_id, conversation_select, rename_input, chat_status]
        new_chat_btn.click(start_new_chat, inputs=[current_user], outputs=new_chat_outputs)
        main_new_chat_btn.click(start_new_chat, inputs=[current_user], outputs=new_chat_outputs)

        conversation_select.change(
            load_conversation_messages,
            inputs=[conversation_select, current_user],
            outputs=[chatbot, active_conversation_id, rename_input, chat_status],
        )

        rename_btn.click(
            rename_conversation,
            inputs=[active_conversation_id, current_user, rename_input],
            outputs=[conversation_select, active_conversation_id, rename_input, chat_status],
        )

        delete_btn.click(
            delete_conversation,
            inputs=[active_conversation_id, current_user],
            outputs=[chatbot, active_conversation_id, conversation_select, rename_input, chat_status],
        )

        send_outputs = [chatbot, message_box, active_conversation_id, conversation_select, rename_input, chat_status]
        send_btn.click(
            send_message,
            inputs=[message_box, chatbot, current_user, active_conversation_id],
            outputs=send_outputs,
        )
        message_box.submit(
            send_message,
            inputs=[message_box, chatbot, current_user, active_conversation_id],
            outputs=send_outputs,
        )
        upload_btn.click(
            upload_document,
            inputs=[file_upload],
            outputs=[upload_status, documents_list],
        )
        save_settings_btn.click(
            update_rag_settings,
            inputs=[num_chunks, show_sources, temperature, chunking_strategy],
            outputs=[settings_status],
        )

    return app


def main() -> None:
    app = create_interface()
    app.queue()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
