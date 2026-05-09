"""
Base abstract class for text chunking strategies.

This module defines the interface that all chunking strategies must implement.
Chunking is a critical step in RAG (Retrieval-Augmented Generation) because:
1. LLMs have context window limits (can't process entire documents at once)
2. Smaller chunks improve semantic search precision
3. Better chunks = better retrieval = better answers

CHUNKING STRATEGIES AVAILABLE IN LLAMA-INDEX:
============================================
LlamaIndex provides pre-built node parsers that you can use directly!
Import from: llama_index.core.node_parser

1. **TokenTextSplitter** (currently used): Fixed token count with overlap
   - Pros: Simple, fast, predictable chunk sizes
   - Cons: May split sentences/paragraphs mid-thought
   - Usage: TokenTextSplitter(chunk_size=500, chunk_overlap=100)
   
2. **SentenceSplitter**: Splits at sentence boundaries, combines to target size
   - Pros: Natural breaks, better readability, respects sentence structure
   - Cons: Variable chunk sizes
   - Usage: SentenceSplitter(chunk_size=1024, chunk_overlap=200)
   - RECOMMENDED: Better than TokenTextSplitter for most use cases
   
3. **SemanticSplitterNodeParser**: Groups text by semantic similarity
   - Pros: Preserves context, natural topic boundaries
   - Cons: More complex, requires embedding model, slower, API costs
   - REQUIRES: Azure OpenAI embeddings configured in .env file
   - REQUIRES: Install `llama-index-embeddings-azure-openai` package
   - Usage: 
     ```python
     from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
     
     # AzureOpenAIEmbedding properly handles Azure OpenAI
     embed_model = AzureOpenAIEmbedding(
         model=os.getenv("AZURE_EMBEDDINGS_MODEL_NAME"),
         deployment_name=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT_NAME"),
         api_key=os.getenv("AZURE_EMBEDDINGS_API_KEY"),
         azure_endpoint=os.getenv("AZURE_EMBEDDINGS_ENDPOINT"),
         api_version=os.getenv("AZURE_EMBEDDINGS_API_VERSION")
     )
     SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model)
     ```
   - Use case: When context preservation is critical (long research papers, technical docs)
   
4. **SentenceWindowNodeParser**: Sliding window with sentence context
   - Pros: Preserves surrounding context, no information loss at boundaries
   - Cons: Redundancy, larger index size
   - Usage: SentenceWindowNodeParser(window_size=3, window_metadata_key="window", original_text_metadata_key="original_text")
   - Use case: When you need context from surrounding sentences
   
5. **HierarchicalNodeParser**: Recursive hierarchical splitting
   - Pros: Respects document structure, creates chunk hierarchy
   - Cons: Requires structured documents, more complex setup
   - Usage: HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
   - Use case: Long documents with clear hierarchical structure

TODO: Replace TokenTextSplitter with another strategy for better results!
      Create new wrapper classes for SemanticSplitterNodeParser and SentenceWindowNodeParser
      if needed.
"""

from abc import ABC, abstractmethod

class ChunkingBase(ABC):
    """
    Abstract base class defining the interface for all chunking strategies.
    
    All chunking implementations must inherit from this class and implement
    the abstract methods to ensure consistent behavior across strategies.
    """

    @abstractmethod
    def __init__(self, embedding_model: str | None):
        """
        Initialize the chunking strategy.
        
        Args:
            embedding_model: Optional model name for embedding-based chunking strategies.
                           For token/sentence-based chunking, this can be None.
                           For semantic chunking (SemanticSplitterNodeParser), this should be
                           configured with Azure OpenAI embeddings from .env file.
        
        Example for semantic chunking with Azure OpenAI:
            ```python
            from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
            import os
            
            # AzureOpenAIEmbedding provides proper Azure OpenAI support
            embed_model = AzureOpenAIEmbedding(
                model=os.getenv("AZURE_EMBEDDINGS_MODEL_NAME"),
                deployment_name=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT_NAME"),
                api_key=os.getenv("AZURE_EMBEDDINGS_API_KEY"),
                azure_endpoint=os.getenv("AZURE_EMBEDDINGS_ENDPOINT"),
                api_version=os.getenv("AZURE_EMBEDDINGS_API_VERSION")
            )
            # Then pass embed_model to SemanticSplitterNodeParser
            ```
        
        TODO: For semantic chunking, store the embedding model to:
              1. Generate embeddings for sentences (using Azure OpenAI)
              2. Calculate similarity between consecutive sentences
              3. Group similar sentences into coherent chunks while similarity is high
        """
        pass

    @abstractmethod
    def _text_splitter(self):
        """
        Internal method to perform the actual text splitting logic.
        
        Returns:
            list: List of text chunks as strings.
        
        Implementation varies by strategy:
        - TokenTextSplitter: Use from llama_index.core.node_parser
        - SentenceSplitter: Use from llama_index.core.node_parser (RECOMMENDED)
        - SemanticSplitterNodeParser: Use from llama_index.core.node_parser
        - SentenceWindowNodeParser: Use from llama_index.core.node_parser
        - HierarchicalNodeParser: Use from llama_index.core.node_parser
        
        All these are pre-built in llama-index - no need to implement from scratch!
        
        TODO: To use a different splitter, simply instantiate it in __init__ and call it here.
              Example for SentenceSplitter:
              ```python
              from llama_index.core.node_parser import SentenceSplitter
              
              def __init__(self):
                  self.splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
              
              def _text_splitter(self):
                  return self.splitter.split_text(self.text)
              ```
        """
        pass
    
    @abstractmethod
    def get_chunks_length(self):
        """
        Return the total number of chunks created from the text.
        
        Returns:
            int: Number of chunks.
        
        Useful for:
        - Debugging chunking behavior
        - Monitoring chunk distribution across documents
        - Estimating storage/embedding costs
        """
        pass
    
    @abstractmethod
    def get_chunks_from_text(self, text: str):
        """
        Main public method to chunk input text.
        
        Args:
            text: The input text to be chunked.
        
        Returns:
            list: List of text chunks.
        
        This is the primary interface used by the ingestion pipeline.
        Should handle:
        1. Store the input text
        2. Call _text_splitter() to perform chunking
        3. Return the resulting chunks
        """
        pass

    @abstractmethod
    def get_metadata(self, node):
        """
        Extract metadata from a chunk/node.
        
        Args:
            node: A chunk node (from llama-index or custom structure).
        
        Returns:
            dict: Metadata dictionary with useful information about the chunk.
        
        Metadata examples:
        - Source document name/path
        - Page number (for PDFs)
        - Section/heading (for structured documents)
        - Chunk position in original document
        - Timestamp (for time-sensitive data)
        
        TODO: Implement this to enable source citation in answers!
              Store metadata alongside embeddings in FAISS to show users
              which documents/pages were used to generate answers.
        
        Example metadata structure:
        {
            "source": "climate_report_2024.pdf",
            "page": 15,
            "chunk_index": 3,
            "section": "Carbon Emissions",
            "char_start": 1024,
            "char_end": 1524
        }
        """
        pass