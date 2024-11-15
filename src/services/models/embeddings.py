from openai import AzureOpenAI
import os

class Embeddings:
    """Handles interactions with the Azure OpenAI Embeddings API.

    Attributes:
        client (AzureOpenAI): The Azure OpenAI client instance.
        model (str): The name of the Azure OpenAI embedding model to use.

    Methods:
        get_embeddings(text): Generates embeddings for the given text using the Azure OpenAI Embeddings API.
    """
    def __init__(self):
        """Initializes the Embeddings class with Azure OpenAI client and model information."""
        azure_endpoint = os.getenv("AZURE_EMBEDDINGS_ENDPOINT")
        azure_deployment = os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT_NAME")
        api_key = os.getenv("AZURE_EMBEDDINGS_API_KEY")
        api_version = os.getenv("AZURE_LLM_API_VERSION")

        self.model = os.getenv("AZURE_EMBEDDINGS_MODEL_NAME")

        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            api_key=api_key
        )



    def get_embeddings(self, text: str) -> list[float]:
        """Generates embeddings for the given text.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            list: A list of floats representing the text embedding.
        """
        completion = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        
        return completion.data[0].embedding
