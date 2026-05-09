"""
Embeddings Service

This module handles text-to-vector conversion using Azure OpenAI's embedding models.
Embeddings are dense vector representations of text that capture semantic meaning,
enabling similarity search in the vector database.

In the RAG architecture, embeddings are used to:
1. Convert document chunks to vectors during ingestion
2. Convert user queries to vectors during retrieval
3. Enable semantic search by comparing vector similarity

The same embedding model must be used for both indexing and querying to ensure
vector compatibility.
"""

from openai import AzureOpenAI
import os


class Embeddings:
    """
    Handles interactions with the Azure OpenAI Embeddings API.
    
    This class provides a simple interface for converting text into high-dimensional
    vector embeddings using Azure OpenAI's embedding models. Embeddings capture the
    semantic meaning of text, allowing for similarity comparisons.
    
    The embedding model (text-embedding-3-large) produces 3072-dimensional vectors.
    These vectors are used by the FAISS index to perform semantic search - finding
    chunks of text that are semantically similar to a query, even if they don't share
    exact keywords.
    
    How it works:
    - "climate change" and "global warming" will have very similar embeddings
    - "climate" and "weather" will have moderately similar embeddings
    - "climate" and "pizza" will have very different embeddings

    Attributes:
        client (AzureOpenAI): The Azure OpenAI client instance for API communication.
        model (str): The name of the embedding model to use (e.g., 'text-embedding-3-large').

    Methods:
        get_embeddings(text): Converts text to a vector embedding.
    
    Example:
        >>> embeddings = Embeddings()
        >>> vector = embeddings.get_embeddings("Climate change is a global issue")
        >>> len(vector)  # 3072 for text-embedding-3-large
        3072
    """
    
    def __init__(self):
        """
        Initializes the Embeddings class with Azure OpenAI client and model information.
        
        Reads configuration from environment variables (.env file):
        - AZURE_EMBEDDINGS_ENDPOINT: The Azure OpenAI service endpoint URL
        - AZURE_EMBEDDINGS_DEPLOYMENT_NAME: The deployment name for the embedding model
        - AZURE_EMBEDDINGS_API_KEY: Authentication key for Azure OpenAI service
        - AZURE_LLM_API_VERSION: API version to use (shared with LLM service)
        - AZURE_EMBEDDINGS_MODEL_NAME: The model name (e.g., 'text-embedding-3-large')
        
        The client is configured once during initialization and reused for all
        embedding requests.
        
        Raises:
            KeyError: If required environment variables are not set.
            openai.OpenAIError: If client initialization fails.
        
        Note:
            The embedding model must match the dimension configured in FAISSIndex.
            text-embedding-3-large outputs 3072-dimensional vectors.
        """
        # Load Azure OpenAI embeddings configuration from environment variables
        azure_endpoint = os.getenv("AZURE_EMBEDDINGS_ENDPOINT")
        azure_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT_NAME")
        api_key = os.getenv("AZURE_EMBEDDINGS_API_KEY")
        api_version = os.getenv("AZURE_LLM_API_VERSION")  # Shared with LLM

        # Store the model name for API calls
        self.model = os.getenv("AZURE_EMBEDDINGS_MODEL_NAME")

        # Initialize the Azure OpenAI client for embeddings
        # This client is separate from the LLM client but uses the same API
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key
        )


    def get_embeddings(self, text: str) -> list[float]:
        """
        Generates embeddings (vector representation) for the given text.
        
        This method sends the text to Azure OpenAI's embedding API and receives back
        a dense vector that represents the semantic meaning of the text. The vector
        can then be stored in FAISS or used for similarity comparisons.
        
        The embedding process:
        1. Text is sent to the Azure OpenAI embedding model
        2. The model processes the text and generates a vector
        3. The vector is returned as a list of floats
        
        The same text will always produce the same embedding (embeddings are deterministic).
        Similar texts will produce similar vectors (close in vector space).

        Args:
            text (str): The text to generate embeddings for. Can be a sentence, paragraph,
                       or document chunk. Longer texts may be truncated by the model
                       (max ~8191 tokens for text-embedding-3-large).

        Returns:
            list[float]: A list of floating-point numbers representing the text embedding.
                        Length is 3072 for text-embedding-3-large.
                        Each number is typically in the range [-1, 1].
        
        Example:
            >>> embeddings = Embeddings()
            >>> vec1 = embeddings.get_embeddings("renewable energy")
            >>> vec2 = embeddings.get_embeddings("solar power")
            >>> # vec1 and vec2 will be similar (close in vector space)
        
        Raises:
            openai.OpenAIError: If the API call fails (network issues, invalid key, etc.)
            openai.RateLimitError: If API rate limits are exceeded
        
        Note:
            Each API call has a cost based on the number of tokens in the input text.
            Consider caching or batching for cost optimization.
        """
        # Call the Azure OpenAI embeddings API
        # The API returns a response object with multiple fields
        completion = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        
        # Extract the embedding vector from the response
        # The response contains metadata and the actual embedding data
        # We only need the vector (list of floats)
        return completion.data[0].embedding
