"""
Large Language Model (LLM) Service

This module handles interactions with Azure OpenAI's GPT models for generating responses
in the RAG (Retrieval-Augmented Generation) chatbot. It manages:
- API client configuration and authentication
- System prompts and conversation history
- Context injection from retrieved documents
- Response generation with streaming support

The LLM is the "G" (Generation) component in RAG, taking retrieved context and
generating coherent, contextual answers to user queries.
"""

from openai import AzureOpenAI
import os


class LLM():
    """
    Handles interactions with the Azure OpenAI LLM (Large Language Model).
    
    This class encapsulates all LLM-related functionality for the chatbot, including:
    - Managing the Azure OpenAI API client
    - Formatting prompts with context and history
    - Generating responses using the configured GPT model
    
    The class uses the OpenAI Python SDK to communicate with Azure-hosted GPT models,
    enabling chat-based interactions with conversation history and context awareness.

    Attributes:
        client (AzureOpenAI): The Azure OpenAI client instance used for API calls.
        model_name (str): The name of the GPT model deployment to use (e.g., 'gpt-4o').

    Methods:
        get_response(history, context, user_input): Generates a response from the LLM
            given conversation history, retrieved context, and the current user input.
    
    Example:
        >>> llm = LLM()
        >>> history = []
        >>> context = "Climate change is caused by greenhouse gas emissions."
        >>> response = llm.get_response(history, context, "What causes climate change?")
        >>> print(response)
    """
    
    def __init__(self):
        """
        Initializes the LLM class with Azure OpenAI client and model information.
        
        Reads configuration from environment variables (.env file):
        - AZURE_LLM_ENDPOINT: The Azure OpenAI service endpoint URL
        - AZURE_LLM_DEPLOYMENT_NAME: The deployment name for the GPT model
        - AZURE_LLM_API_KEY: Authentication key for Azure OpenAI service
        - AZURE_LLM_API_VERSION: API version to use (e.g., '2024-12-01-preview')
        - AZURE_LLM_MODEL_NAME: The model name (e.g., 'gpt-4o')
        
        The AzureOpenAI client is configured with these credentials and will be used
        for all subsequent API calls.
        
        Raises:
            KeyError: If required environment variables are not set.
            openai.OpenAIError: If client initialization fails (e.g., invalid credentials).
        
        Note:
            Make sure the .env file is loaded before instantiating this class.
            The API key should be kept secure and never committed to version control.
        """
        # Load Azure OpenAI configuration from environment variables
        azure_endpoint = os.getenv("AZURE_LLM_ENDPOINT")
        azure_deployment = os.getenv("AZURE_LLM_DEPLOYMENT_NAME")
        api_key = os.getenv("AZURE_LLM_API_KEY")
        api_version = os.getenv("AZURE_LLM_API_VERSION")

        # Initialize the Azure OpenAI client with configuration
        # This client will handle authentication and API communication
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key
        )
        
        # Store the model name for use in API calls
        self.model_name = os.getenv("AZURE_LLM_MODEL_NAME")


    def get_response(self, history, context, user_input):
        """
        Generates a response from the LLM based on conversation history, context, and user input.
        
        This is the core method of the RAG system's generation component. It:
        1. Formats the system prompt to guide LLM behavior
        2. Injects retrieved context (from FAISS) into the user's message
        3. Includes conversation history for context awareness
        4. Calls the OpenAI API to generate a response
        
        The context parameter contains relevant document chunks retrieved from the
        vector database, which the LLM uses to ground its response in factual information.

        Args:
            history (list): A list of previous messages in the conversation.
                          Each message is a dict with 'role' and 'content' keys.
                          Roles can be 'user' or 'assistant'.
                          Example: [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
            
            context (str): Relevant information from the knowledge base (retrieved chunks).
                         This is concatenated text from the most similar document chunks,
                         separated by "#####" delimiters. This context is hidden from the user
                         but visible to the LLM for generating accurate answers.
            
            user_input (str): The user's current question or message.
                            This will be combined with the context before sending to the LLM.

        Returns:
            str: The LLM's generated response as a string. This is the answer that will be
                shown to the user in the chatbot interface.
        
        Example:
            >>> history = [{"role": "user", "content": "What is CO2?"}, 
            ...           {"role": "assistant", "content": "CO2 is carbon dioxide..."}]
            >>> context = "Carbon dioxide is a greenhouse gas..."
            >>> user_input = "What causes CO2 emissions?"
            >>> response = llm.get_response(history, context, user_input)
        
        TODO: Improve system prompt - current prompt is very basic
              Consider adding:
              - Specific instructions for climate change domain
              - Guidelines for citing sources
              - Instructions for recommending sustainable products when appropriate
              - Tone and style guidelines (e.g., educational, friendly)
              - Instructions to decline answering off-topic questions
        
        TODO: Add source attribution in responses - modify prompt to ask LLM to cite sources
        
        TODO: Implement product recommendation logic - detect when user wants product suggestions
              and use the CSV data to recommend sustainable products
                        
        TODO: Add safety filters - ensure responses don't contain harmful content
        """
        # TODO: Improve system prompt - current implementation is basic
        # The system prompt defines the AI's role and behavior
        # Current prompt just tells the AI to use the provided context
        # Consider expanding this to:
        # - Define the AI as a climate change expert
        # - Add instructions for source citation
        # - Add product recommendation capabilities
        # - Set tone and personality
        SYSTEM_PROMPT = "Answer based on the context provided. The user does not have visibility on this context."
        
        # TODO: Enhanced system prompt example:
        # SYSTEM_PROMPT = """You are EcoGuide, an AI assistant specialized in climate change and sustainability.
        # Answer questions based on the provided context from scientific reports and articles.
        # 
        # Guidelines:
        # - Provide accurate, educational responses about climate change
        # - When appropriate, recommend sustainable products from the product database
        # - Cite sources when making factual claims
        # - If the answer isn't in the context, politely say so
        # - Maintain a friendly, encouraging tone
        # 
        # The user cannot see the context provided to you - it's for your reference only."""

        # Combine user input with retrieved context
        # The context is prepended to the user's question, hidden from their view
        # Format: 
        #   #CONTEXT
        #   <retrieved chunks separated by #####>
        #   #USER INPUT
        #   <actual user question>
        user_input_with_context = f"#CONTEXT\n{context}\n#USER INPUT\n{user_input}"
        
        # Construct the full message list for the API call
        # Messages are sent in order: system prompt, history, current message
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,  # Unpack conversation history (previous turns)
            {"role": "user", "content": user_input_with_context}
        ]

        # Call the Azure OpenAI API to generate a response
        # stream=False means we wait for the complete response (not token-by-token)
        # TODO: Set stream=True for better UX with real-time response display
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False
        )

        # Extract and return the text content of the response
        # The API returns a complex object; we just need the message content
        return response.choices[0].message.content
