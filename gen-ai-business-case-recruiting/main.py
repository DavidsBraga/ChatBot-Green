"""
EcoGuide Chatbot - Main Application Entry Point

This is the main script for running the RAG-based climate change chatbot for the
Deloitte Tech Experience Bootcamp challenge. The chatbot answers questions about
climate change using a knowledge base of scientific documents and can recommend
sustainable products.

Architecture:
- RAG (Retrieval-Augmented Generation) pattern
- FAISS vector database for semantic search
- Azure OpenAI for embeddings and LLM
- Terminal-based interface (Gradio frontend to be implemented)

Usage:
    python main.py

The chatbot will:
1. Load environment variables from .env
2. Initialize the LLM and embedding models
3. Load or create the FAISS index from documents
4. Enter an interactive loop for user queries
5. Type 'exit' to quit

TODO: Implement Gradio web interface to replace terminal interaction
TODO: Add source citation in responses
TODO: Implement CSV-based product recommendations
TODO: Implement multimodal RAG for image processing
"""

import faiss
from openai import AzureOpenAI
from src.services.models.embeddings import Embeddings
from src.services.vectorial_db.faiss_index import FAISSIndex
from src.ingestion.ingest_files import ingest_files_data_folder
from src.services.models.llm import LLM
import os
from dotenv import load_dotenv
import time


def chatbot(llm: LLM, input_text: str, history: list, index: FAISSIndex):
    """
    Processes a user query through the RAG pipeline and returns a response.
    
    This function implements the complete RAG workflow:
    1. RETRIEVAL: Use the FAISS index to find relevant document chunks
    2. AUGMENTATION: Combine retrieved chunks into context
    3. GENERATION: Use LLM to generate an answer based on context
    
    The function also manages conversation history and provides timing information
    for performance monitoring.

    Args:
        llm (LLM): An instance of the LLM class for generating responses.
        input_text (str): The user's input question or message.
        history (list): A list of previous messages in the conversation history.
                       Each message is a dict with 'role' and 'content' keys.
        index (FAISSIndex): An instance of the FAISSIndex class for retrieving
                           relevant information from the knowledge base.

    Returns:
        tuple: A tuple containing two elements:
            - response (str): The AI's generated response to the user's query
            - history (list): The updated conversation history including the new
                            user message and AI response
    
    TODO: Add source citation - track which documents were used
    """
    # STEP 1: RETRIEVAL - Find relevant document chunks
    start = time.time()
    retrieved_chunks = index.retrieve_chunks(input_text, num_chunks=5)
    
    # Combine retrieved chunks into a single context string
    # Chunks are separated by ##### for clarity in the prompt
    # TODO: Add source attribution - include document names with each chunk
    context = "\n\n#####\n\n".join(retrieved_chunks)
    
    print("Time for retrieval =", time.time() - start, "seconds")
    
    # STEP 2: GENERATION - Use LLM to generate answer with context
    start = time.time()
    response = llm.get_response(history, context, input_text)
    print("Time for response =", time.time() - start, "seconds")

    # STEP 3: HISTORY MANAGEMENT - Update conversation history
    # TODO: Consider limiting history length to avoid token limits
    # TODO: Implement conversation summarization for long chats
    history.append({"role": "user", "content": input_text})
    history.append({"role": "assistant", "content": response})
    
    return response, history


def main():
    """
    Main function to run the chatbot application.
    
    This function orchestrates the entire chatbot lifecycle:
    1. Initialize Azure OpenAI clients (LLM and embeddings)
    2. Create or load the FAISS vector index
    3. Run the interactive chat loop
    
    The function handles:
    - Environment setup
    - Model initialization
    - Index loading/creation
    - User interaction loop
    - Graceful shutdown
    
    TODO: Replace terminal loop with Gradio web interface
    TODO: Add error handling
    TODO: Implement logging for monitoring and debugging
    """
    # Initialize the LLM (GPT-4o) for response generation
    llm = LLM()

    # Initialize the embeddings model (text-embedding-3-large) for vector conversion
    # Note: Dimension is 3072 for text-embedding-3-large
    embeddings = Embeddings()
    index = FAISSIndex(embeddings=embeddings.get_embeddings, dimension=3072)

    # Try to load existing index from disk, or create new one if not found
    try:
        # Loading existing index avoids re-processing all documents and API costs
        index.load_index()
    except FileNotFoundError:
        # First time setup: ingest all documents from the data folder
        print("\nNo existing index found. Creating new index from documents...")
        print("This may take a few minutes on first run...\n")
        ingest_files_data_folder(index)
        
        # Save the index for future runs
        index.save_index()
        print("\nIndex created and saved successfully!\n")

    # Initialize empty conversation history
    history = []
    
    print("\n# INITIALIZED CHATBOT #")
    print("Ask me anything about climate change!")
    print("Type 'exit' to quit.\n")
    
    # TODO: Replace this terminal loop with Gradio web interface
    # The Gradio interface should include:
    # - Chat interface with message history
    # - Source citation display
    # - Product recommendation cards
    # - Export conversation button
    # - File upload for adding documents (optional)
    
    # Main interaction loop
    while True:
        # Get user input from terminal
        # TODO: Replace with Gradio chat interface
        user_input = str(input("You:  "))
        
        # Check for exit command
        if user_input.lower() == "exit":
            print("\nThank you for using EcoGuide! Goodbye!")
            break
        
        # Process the query through the RAG pipeline
        response, history = chatbot(llm, user_input, history, index)
        
        # Display the response
        # TODO: In Gradio, this would be rendered as a chat message
        # TODO: Add source citations below the response
        print("AI: ", response)
        print()  # Blank line for readability


if __name__ == "__main__":
    # Load environment variables from .env file
    # This must be done before importing any modules that use environment variables
    load_dotenv(override=True)
    
    # Run the main chatbot function
    main()
