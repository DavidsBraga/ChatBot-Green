"""
Gradio Web Interface for EcoGuide Chatbot

This module provides a web-based user interface for the RAG chatbot using Gradio.
Gradio is a Python library that makes it easy to create web UIs for ML models.

TODO: Complete the implementation of this Gradio interface to replace the terminal-based
      interaction in main.py. This will provide a better user experience with:
      - Chat history display
      - Source citation display
      - Product recommendations
      - File upload for new documents
      - Conversation export

REQUIREMENTS:
- Install gradio: pip install gradio
- Use the existing RAG components (FAISSIndex, LLM, EmbeddingsService)
- Maintain conversation history per session
- Display retrieved sources with each answer

GRADIO DOCUMENTATION:
- Main docs: https://www.gradio.app/docs
- Chat interface: https://www.gradio.app/docs/chatinterface
- Blocks API: https://www.gradio.app/docs/blocks
"""

import gradio as gr
from dotenv import load_dotenv
import os

# TODO: Import the necessary components from your RAG system
# Hint: You'll need FAISSIndex, LLM, and possibly EmbeddingsService
# from src.services.vectorial_db.faiss_index import FAISSIndex
# from src.services.models.llm import LLM
# from src.services.models.embeddings import EmbeddingsService
# from src.ingestion.ingest_files import ingest_files_data_folder


# Load environment variables
load_dotenv(override=True)


# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

# TODO: Initialize your RAG components globally
# These should be initialized once when the app starts, not on every request
# 
# Example:
# embeddings_service = EmbeddingsService()
# faiss_index = FAISSIndex(dimension=3072, embeddings=embeddings_service)
# llm = LLM()
#
# # Load existing index or ingest documents
# try:
#     faiss_index.load_index()
#     print("✅ Loaded existing FAISS index")
# except:
#     print("📁 No existing index found. Ingesting documents...")
#     ingest_files_data_folder(faiss_index)
#     faiss_index.save_index()
#     print("✅ Documents ingested and index saved")


# TODO: Initialize global variables for state management
# conversation_histories = {}  # Dictionary to store conversation history per session


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_sources(retrieved_chunks, num_sources=3):
    """
    Format retrieved document chunks as source citations.
    
    TODO: Implement source formatting to show which documents were used.
    
    Args:
        retrieved_chunks (list): List of text chunks retrieved from FAISS
        num_sources (int): Number of sources to display
    
    Returns:
        str: Formatted source citations in HTML or Markdown
    
    Example output:
        **Sources:**
        1. climate_report_2024.pdf (page 15)
        2. greenhouse_protocol_FAQ.html (section 3)
        3. sustainable_products.csv (row 45)
    
    HINT: Currently FAISS only returns text chunks without metadata.
          You need to modify faiss_index.py to store and return metadata
          (document name, page number, etc.) alongside chunks.
    """
    # TODO: Implement source citation formatting
    # For now, return a placeholder
    return "\n\n**Sources:** _Not yet implemented - see faiss_index.py TODO for metadata tracking_"


def extract_product_recommendations(response_text):
    """
    Extract product recommendations from the LLM response.
    
    TODO: Implement product recommendation extraction and formatting.
    
    Args:
        response_text (str): The LLM's response text
    
    Returns:
        str: Formatted product recommendations in HTML/Markdown, or empty string
    
    APPROACH 1: Parse LLM response for product mentions
    - Look for product names/IDs in the response
    - Match against sustainable_products.csv
    - Format as a nice product card
    
    APPROACH 2: Separate product search
    - Use an LLM to detect if user is asking for products
    - Provide the LLM with the user's query and the CSV data and ask for recommendations
    - Format results as product recommendations
    
    HINT: You'll need to load sustainable_products.csv and search it
          based on the user's query or LLM response.
    """
    # TODO: Implement product recommendation logic
    return ""


def chatbot_response(message, history):
    """
    Main chatbot function that processes user input and returns a response.
    
    This function implements the RAG workflow:
    1. Retrieve relevant chunks from FAISS
    2. Augment the query with retrieved context
    3. Generate response using LLM
    4. Format the response with sources and recommendations
    
    Args:
        message (str): The user's input message
        history (list): Chat history in Gradio format [(user_msg, bot_msg), ...]
    
    Returns:
        str: The chatbot's response with sources and recommendations
    
    TODO: Implement the complete RAG pipeline here
    """
    # TODO: Step 1 - Convert Gradio history format to LLM format
    # Gradio history: [(user_msg, bot_msg), (user_msg, bot_msg), ...]
    # LLM expects: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    #
    # llm_history = []
    # for user_msg, bot_msg in history:
    #     llm_history.append({"role": "user", "content": user_msg})
    #     if bot_msg:  # bot_msg might be None for the current message
    #         llm_history.append({"role": "assistant", "content": bot_msg})
    
    # TODO: Step 2 - Retrieve relevant chunks from FAISS
    # retrieved_chunks = faiss_index.retrieve_chunks(message, num_chunks=5)
    # context = "\n\n#####\n\n".join(retrieved_chunks)
    
    # TODO: Step 3 - Generate response using LLM with context
    # response = llm.get_response(llm_history, context, message)
    
    # TODO: Step 4 - Add source citations
    # sources = format_sources(retrieved_chunks)
    # formatted_response = f"{response}\n\n---\n{sources}"
    
    # TODO: Step 5 - Check for and add product recommendations
    # products = extract_product_recommendations(response)
    # if products:
    #     formatted_response += f"\n\n{products}"
    
    # TODO: Return the complete formatted response
    # return formatted_response
    
    # PLACEHOLDER: Remove this once you implement the above
    return f"🚧 **TODO**: Implement chatbot_response() function\n\nYou said: {message}\n\nThis is where the RAG pipeline should process your question and return an answer with sources."


def reset_conversation():
    """
    Reset the conversation history.
    
    TODO: Implement conversation reset functionality.
    
    Returns:
        tuple: Empty history and a message confirming reset
    """
    # TODO: Clear the conversation history
    # return [], "Conversation reset! Ask me anything about climate change."
    
    return [], "Conversation reset!"


def export_conversation(history):
    """
    Export the conversation history to a file.
    
    TODO: Implement conversation export functionality.
    
    Args:
        history (list): Chat history in Gradio format
    
    Returns:
        str: Path to the exported file, or error message
    
    HINT: You can export as:
    - Plain text (.txt)
    - JSON (.json) 
    - Markdown (.md)
    - PDF (requires reportlab or similar)
    """
    # TODO: Implement conversation export
    # Example:
    # import json
    # export_data = {
    #     "timestamp": datetime.now().isoformat(),
    #     "conversation": history
    # }
    # filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # with open(filename, 'w') as f:
    #     json.dump(export_data, f, indent=2)
    # return filename
    
    return "🚧 TODO: Implement export_conversation() function"


def upload_document(file):
    """
    Upload and ingest a new document into the RAG system.
    
    TODO: Implement document upload and ingestion.
    
    Args:
        file: File object from Gradio file upload component
    
    Returns:
        str: Status message about the upload
    
    WORKFLOW:
    1. Save the uploaded file to a temporary location
    2. Determine file type (PDF, HTML, DOCX, etc.)
    3. Load the document using appropriate loader
    4. Chunk the document text
    5. Generate embeddings and add to FAISS index
    6. Save the updated index
    7. Return success message
    
    HINT: You can reuse code from ingest_files.py
    """
    # TODO: Implement document upload and ingestion
    # import tempfile
    # import shutil
    # from src.ingestion.loaders.loader import Loader
    # from src.ingestion.chunking.token_chunking import text_to_chunks
    #
    # # Save uploaded file
    # temp_path = tempfile.mktemp(suffix=os.path.splitext(file.name)[1])
    # shutil.copy(file.name, temp_path)
    #
    # # Load and process
    # loader = Loader()
    # text = loader.load(temp_path)
    # chunks = text_to_chunks(text)
    #
    # # Add to index
    # faiss_index.ingest_text(text_chunks=chunks)
    # faiss_index.save_index()
    #
    # return f"✅ Successfully ingested {file.name}"
    
    return "🚧 TODO: Implement upload_document() function"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """
    Create and configure the Gradio interface.
    
    TODO: Build a complete Gradio interface with multiple tabs/sections.
    
    RECOMMENDED STRUCTURE:
    
    Tab 1: Chat Interface
    - Chat history display (scrollable)
    - Message input box
    - Submit button
    - Clear conversation button
    - Export conversation button
    
    Tab 2: Document Management
    - File upload component
    - List of currently indexed documents
    - Re-index button
    
    Tab 3: Settings (Optional)
    - Number of chunks to retrieve (slider)
    - Temperature for LLM (slider)
    - Enable/disable source citations (checkbox)
    - Chunking strategy selection (dropdown)
    
    GRADIO COMPONENTS TO USE:
    - gr.ChatInterface: Pre-built chat interface (easiest)
    - gr.Blocks: Custom layout with more control
    - gr.Textbox: For message input
    - gr.Chatbot: For displaying chat history
    - gr.Button: For actions
    - gr.File: For file uploads
    - gr.Slider, gr.Checkbox, gr.Dropdown: For settings
    """
    
    # TODO: OPTION 1 - Simple Chat Interface (Recommended for MVP)
    # This is the quickest way to get started
    #
    # interface = gr.ChatInterface(
    #     fn=chatbot_response,
    #     title="🌍 EcoGuide - Climate Change Chatbot",
    #     description="Ask me anything about climate change, carbon emissions, or sustainable products!",
    #     examples=[
    #         "What is climate change?",
    #         "How can I reduce my carbon footprint?",
    #         "What are carbon offset strategies?",
    #         "Recommend sustainable products for my home",
    #         "Explain the greenhouse gas protocol"
    #     ],
    #     theme=gr.themes.Soft(),
    #     retry_btn="🔄 Retry",
    #     undo_btn="↩️ Undo",
    #     clear_btn="🗑️ Clear",
    # )
    
    # TODO: OPTION 2 - Advanced Interface with Blocks (More Features)
    # Use this for a more customized interface with multiple features
    #
    # with gr.Blocks(theme=gr.themes.Soft(), title="EcoGuide Chatbot") as interface:
    #     gr.Markdown("# 🌍 EcoGuide - Your Climate Change Assistant")
    #     gr.Markdown("Ask questions about climate change and get AI-powered answers with sources!")
    #     
    #     with gr.Tab("💬 Chat"):
    #         chatbot = gr.Chatbot(
    #             height=500,
    #             show_label=False,
    #             avatar_images=(None, "🌍")  # User, Bot avatars
    #         )
    #         msg = gr.Textbox(
    #             placeholder="Type your question here...",
    #             show_label=False,
    #             container=False
    #         )
    #         
    #         with gr.Row():
    #             submit = gr.Button("Send 📤", variant="primary")
    #             clear = gr.Button("Clear 🗑️")
    #             export = gr.Button("Export 💾")
    #         
    #         # Examples
    #         gr.Examples(
    #             examples=[
    #                 "What is climate change?",
    #                 "How can I reduce my carbon footprint?",
    #                 "What are carbon offset strategies?",
    #                 "Recommend sustainable products"
    #             ],
    #             inputs=msg
    #         )
    #         
    #         # Wire up the interactions
    #         msg.submit(chatbot_response, [msg, chatbot], [chatbot])
    #         submit.click(chatbot_response, [msg, chatbot], [chatbot])
    #         clear.click(reset_conversation, None, [chatbot, msg])
    #         export.click(export_conversation, [chatbot], [gr.Textbox(label="Export Path")])
    #     
    #     with gr.Tab("📁 Documents"):
    #         gr.Markdown("### Upload New Documents")
    #         file_upload = gr.File(
    #             label="Upload PDF, HTML, DOCX, or CSV files",
    #             file_types=[".pdf", ".html", ".docx", ".csv"]
    #         )
    #         upload_btn = gr.Button("Upload & Ingest 📤", variant="primary")
    #         upload_status = gr.Textbox(label="Status", interactive=False)
    #         
    #         upload_btn.click(upload_document, [file_upload], [upload_status])
    #         
    #         gr.Markdown("### Current Documents")
    #         gr.Markdown("- Greenhouse_ga_protocol_corporate_standard_FAQ.html")
    #         gr.Markdown("- sustainable_products.csv")
    #         gr.Markdown("_TODO: Make this dynamic by reading from data folder_")
    #     
    #     with gr.Tab("⚙️ Settings"):
    #         gr.Markdown("### RAG Settings")
    #         
    #         num_chunks = gr.Slider(
    #             minimum=1,
    #             maximum=10,
    #             value=5,
    #             step=1,
    #             label="Number of chunks to retrieve",
    #             info="More chunks = more context but slower"
    #         )
    #         
    #         show_sources = gr.Checkbox(
    #             label="Show source citations",
    #             value=True,
    #             info="Display which documents were used for the answer"
    #         )
    #         
    #         chunking_strategy = gr.Dropdown(
    #             choices=["token", "sentence", "semantic"],
    #             value="token",
    #             label="Chunking Strategy",
    #             info="How to split documents (requires re-ingestion to apply)"
    #         )
    #         
    #         gr.Markdown("_TODO: Wire up these settings to actually affect the RAG pipeline_")
    
    # PLACEHOLDER: Simple interface until you implement one of the above
    interface = gr.Interface(
        fn=lambda x: "🚧 **Not Implemented Yet**\n\nTODO: Complete the Gradio interface implementation.\n\nSee gradio_app.py for detailed instructions.",
        inputs=gr.Textbox(label="Your Question", placeholder="Ask about climate change..."),
        outputs=gr.Textbox(label="Response"),
        title="🌍 EcoGuide - Climate Change Chatbot",
        description="⚠️ Under Construction - See gradio_app.py to implement the full interface",
        examples=[
            "What is climate change?",
            "How can I reduce my carbon footprint?",
            "Recommend sustainable products"
        ]
    )
    
    return interface


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to launch the Gradio app.
    
    TODO: Configure launch parameters for production deployment.
    """
    print("=" * 80)
    print("🌍 EcoGuide Chatbot - Gradio Web Interface")
    print("=" * 80)
    
    # TODO: Uncomment and configure launch parameters
    # interface = create_interface()
    # interface.launch(
    #     server_name="0.0.0.0",  # Listen on all network interfaces
    #     server_port=7860,        # Port number
    #     share=False,             # Set True to create public link (via Gradio servers)
    #     debug=True,              # Enable debug mode for development
    #     show_error=True          # Show detailed errors
    # )
    
    # PLACEHOLDER: Show instructions
    print("\n⚠️  Gradio interface not yet implemented!")
    print("\nTODO List:")
    print("1. Uncomment and initialize RAG components (FAISSIndex, LLM)")
    print("2. Implement chatbot_response() function with RAG pipeline")
    print("3. Implement format_sources() for source citation")
    print("4. Implement extract_product_recommendations() for product suggestions")
    print("5. Create Gradio interface using gr.ChatInterface or gr.Blocks")
    print("6. Wire up all event handlers (submit, clear, export, upload)")
    print("7. Test the interface locally")
    print("8. Deploy (optional: use Gradio share or deploy to Hugging Face Spaces)")
    print("\nFor help, see:")
    print("- Gradio docs: https://www.gradio.app/docs")
    print("- Example in gradio_app.py (this file)")
    print("=" * 80)


if __name__ == "__main__":
    # TODO: Remove this check once you implement the interface
    print("\n" + "!" * 80)
    print("WARNING: This is a template file with TODOs for implementation")
    print("!" * 80 + "\n")
    
    main()


# ============================================================================
# ADDITIONAL IMPLEMENTATION HINTS
# ============================================================================

"""
DEPLOYMENT OPTIONS:
===================

1. **Local Development**:
   python gradio_app.py
   # Access at http://localhost:7860

GRADIO FEATURES TO EXPLORE:
============================

1. **Theming**: Use gr.themes.Soft(), gr.themes.Base(), or create custom theme
2. **Authentication**: Add login with gr.Interface(..., auth=("username", "password"))
3. **Queue**: Handle multiple users with queue=True
4. **Flagging**: Let users flag good/bad responses for improvement
5. **State**: Use gr.State() to maintain user session data
6. **Layout**: Use gr.Row(), gr.Column() for custom layouts
7. **Markdown**: Rich formatting with gr.Markdown()
8. **Analytics**: Track usage with flagging or custom logging

TESTING CHECKLIST:
==================

Before deploying, test:
- [ ] Basic chat functionality works
- [ ] Source citations are displayed correctly
- [ ] Product recommendations appear when relevant
- [ ] Conversation can be cleared
- [ ] Conversation can be exported
- [ ] New documents can be uploaded and ingested
- [ ] Error handling (what if FAISS index is empty?)
- [ ] Mobile responsiveness (Gradio handles this automatically)
- [ ] Multiple concurrent users (use queue=True)


PERFORMANCE OPTIMIZATION:
=========================

1. **Lazy Loading**: Only load FAISS index when first needed
2. **Caching**: Cache frequently asked questions
3. **Async**: Use async/await for non-blocking operations
4. **Streaming**: Stream LLM responses for better UX (requires LLM streaming support)
5. **Batch Processing**: Process multiple user queries in batches if possible


SECURITY CONSIDERATIONS:
========================

1. **Input Validation**: Sanitize user inputs to prevent injection attacks
2. **Rate Limiting**: Prevent abuse by limiting requests per user
3. **File Upload**: Validate file types and sizes to prevent malicious uploads

Good luck with the implementation! 🚀
"""
