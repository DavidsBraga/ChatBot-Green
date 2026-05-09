"""
Token-based text chunking implementation.

This module implements a simple token-based chunking strategy where text is split
into fixed-size chunks based on token count, with configurable overlap between chunks.

TOKEN CHUNKING EXPLAINED:
========================
- Tokens are word-pieces used by language models (not just words!)
- Example: "sustainability" might be 1 token, "climate-change" might be 2 tokens
- Chunk size of 500 tokens ≈ 375 words ≈ 1-2 paragraphs of text
- Overlap ensures context isn't lost at chunk boundaries

CURRENT CONFIGURATION:
=====================
- Chunk size: 500 tokens (~375 words)
- Overlap: 100 tokens (~75 words)
- This means consecutive chunks share 100 tokens for continuity

ADVANTAGES:
- Simple and fast
- Predictable chunk sizes (important for embedding limits)
- Works with all text types (no structure required)

LIMITATIONS:
- May split sentences mid-thought
- Doesn't respect document structure (headings, sections)
- No semantic awareness (related content might be separated)

BETTER ALTERNATIVES IN LLAMA-INDEX:
===================================
Instead of implementing chunking strategies by hand, use llama-index's pre-built node parsers!

1. **SentenceSplitter** (RECOMMENDED upgrade from TokenTextSplitter):
   - Splits at sentence boundaries, then combines to target size
   - Much better than token-based for preserving context
   - Usage:
   ```python
   from llama_index.core.node_parser import SentenceSplitter
   splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
   chunks = splitter.split_text(text)
   ```

2. **SemanticSplitterNodeParser** (for topic-based chunking):
   - Automatically detects topic changes using embeddings
   - Groups semantically similar sentences together
   - REQUIRES: Azure OpenAI embeddings (configured in .env file)
   - REQUIRES: Install `llama-index-embeddings-azure-openai` package
   - Usage:
   ```python
   from llama_index.core.node_parser import SemanticSplitterNodeParser
   from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
   import os
   
   # AzureOpenAIEmbedding properly handles Azure OpenAI
   embed_model = AzureOpenAIEmbedding(
       model=os.getenv("AZURE_EMBEDDINGS_MODEL_NAME"),
       deployment_name=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT_NAME"),
       api_key=os.getenv("AZURE_EMBEDDINGS_API_KEY"),
       azure_endpoint=os.getenv("AZURE_EMBEDDINGS_ENDPOINT"),
       api_version=os.getenv("AZURE_EMBEDDINGS_API_VERSION"),
   )
   
   splitter = SemanticSplitterNodeParser(
       buffer_size=1,
       breakpoint_percentile_threshold=95,
       embed_model=embed_model  # Azure OpenAI embeddings
   )
   nodes = splitter.get_nodes_from_documents([Document(text=text)])
   ```

3. **SentenceWindowNodeParser** (for sliding window with context):
   - Keeps surrounding sentences as metadata for each chunk
   - Useful when retrieval needs broader context
   - Usage:
   ```python
   from llama_index.core.node_parser import SentenceWindowNodeParser
   
   splitter = SentenceWindowNodeParser(
       window_size=3,  # 3 sentences before and after
       window_metadata_key="window",
       original_text_metadata_key="original_text"
   )
   nodes = splitter.get_nodes_from_documents([Document(text=text)])
   ```

4. **HierarchicalNodeParser** (for multi-level chunking):
   - Creates chunks at multiple granularity levels
   - Useful for long documents with clear structure
   - Usage:
   ```python
   from llama_index.core.node_parser import HierarchicalNodeParser
   
   splitter = HierarchicalNodeParser.from_defaults(
       chunk_sizes=[2048, 512, 128]  # Three levels of granularity
   )
   nodes = splitter.get_nodes_from_documents([Document(text=text)])
   ```

TODO: Consider upgrading to another chunking strategy for better chunking quality!
"""

from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Document
from src.ingestion.chunking.chunking_base import ChunkingBase
from llama_index.core import SimpleDirectoryReader

class TokenChunking(ChunkingBase):
    """
    Token-based chunking strategy implementation.
    
    Splits text into chunks of fixed token count with overlap between consecutive chunks.
    Uses llama-index's TokenTextSplitter which handles token counting internally.
    """
    
    def __init__(self):
        """
        Initialize token chunking with default parameters.
        
        Note: This implementation doesn't use embedding_model parameter from base class
        because token chunking doesn't need embeddings (unlike semantic chunking).
        
        TODO: Consider making chunk_size and chunk_overlap configurable via __init__ params
              to allow different chunking strategies for different document types:
              - Technical docs: Larger chunks (800 tokens) for more context
              - FAQ content: Smaller chunks (300 tokens) for precise answers
              - Product descriptions: Medium chunks (500 tokens)
        """
        self.DEFAULT_CHUNK_SIZE = 500     # Target tokens per chunk
        self.DEFAULT_CHUNK_OVERLAP = 100  # Tokens shared between consecutive chunks
    
    def _text_splitter(self):
        """
        Internal method that performs the actual token-based splitting.
        
        Returns:
            list[str]: List of text chunks.
        
        How it works:
        1. TokenTextSplitter counts tokens in the input text
        2. Splits text at ~500 token boundaries
        3. Includes 100 token overlap from previous chunk
        4. Returns list of chunk strings
        
        Example with 1200 tokens:
        - Chunk 1: tokens 0-500
        - Chunk 2: tokens 400-900 (100 overlap with chunk 1)
        - Chunk 3: tokens 800-1200 (100 overlap with chunk 2)
        
        TODO: Add error handling for edge cases:
              - Empty text input
              - Text shorter than chunk_size
              - Very long documents (might need batching)
        """
        print("Running token chunker...")        
        splitter = TokenTextSplitter(
            chunk_size=self.DEFAULT_CHUNK_SIZE,
            chunk_overlap=self.DEFAULT_CHUNK_OVERLAP
        )
        chunks = splitter.split_text(self.text)
        return chunks
    
    def get_chunks_length(self):
        """
        Returns the number of chunks created from the text.
        
        Returns:
            int: Total number of chunks.
        
        Useful for debugging and monitoring:
        - Check if chunking is producing expected number of chunks
        - Estimate embedding API costs (1 API call per chunk)
        - Monitor chunk distribution across different document types
        """
        return len(self.chunks)
    
    def get_chunks_from_text(self, text: str) -> list:
        """
        Main public method to chunk input text.
        
        Args:
            text: The input text string to be chunked.
        
        Returns:
            list[str]: List of text chunks.
        
        This is called by the ingestion pipeline (faiss_index.py) to convert
        raw document text into chunks before embedding generation.
        
        Workflow:
        1. Store text as instance variable (for get_chunks_length)
        2. Call _text_splitter() to perform chunking
        3. Store and return resulting chunks
        
        TODO: Add preprocessing before chunking:
              - Remove excessive whitespace/newlines
              - Normalize special characters
              - Handle different encodings (UTF-8, Latin-1, etc.)
        """
        self.text = text
        self.chunks = self._text_splitter()
        return self.chunks
    
    def get_metadata(self, node):
        """
        Extract metadata from a chunk node.
        
        Args:
            node: A llama-index Node object containing chunk data and metadata.
        
        Returns:
            dict: Metadata information about the chunk.
        
        NOTE: Currently not implemented (raises NotImplementedError).
        
        TODO: Implement this method to enable SOURCE CITATION feature!
        
        Implementation example:
        ```python
        def get_metadata(self, node):
            return {
                "source_file": node.metadata.get("file_name", "unknown"),
                "page_number": node.metadata.get("page", None),
                "chunk_index": node.metadata.get("chunk_id", 0),
                "start_char": node.start_char_idx,
                "end_char": node.end_char_idx,
                "chunk_size": len(node.text)
            }
        ```
        
        This metadata should be stored alongside embeddings in FAISS so that
        when chunks are retrieved, the chatbot can cite which documents/pages
        were used to generate the answer.
        
        Example user experience:
        User: "What are carbon offset strategies?"
        Bot: "Carbon offset strategies include reforestation and renewable energy 
              projects... (Source: climate_report_2024.pdf, page 15)"
        """
        raise NotImplementedError

def text_to_chunks(text: str, chunk_size: int = 512) -> list[str]:
    """
    Convenience function to quickly chunk text using token-based strategy.
    
    Args:
        text: The text to split into chunks.
        chunk_size: The desired size of each chunk (default is 512 words).
                   NOTE: This parameter is currently IGNORED - the function uses
                   TokenChunking's hardcoded DEFAULT_CHUNK_SIZE (500 tokens).
    
    Returns:
        list[str]: A list of text chunks.
    
    This is a simple wrapper around TokenChunking class for quick use.
    Used primarily by the FAISS index ingestion pipeline.
    
    TODO: Fix parameter usage - actually pass chunk_size to TokenChunking:
    ```python
    def text_to_chunks(text: str, chunk_size: int = 512) -> list[str]:
        chunker = TokenChunking(chunk_size=chunk_size)  # Pass parameter
        chunks = chunker.get_chunks_from_text(text)
        return chunks
    ```
    
    TODO: Consider upgrading to a better chunking strategy using llama-index pre-built parsers.
          See module docstring above for available options (SentenceSplitter, SemanticSplitterNodeParser, etc.)
    """
    chunker = TokenChunking()
    chunks = chunker.get_chunks_from_text(text)
    return chunks


# ============================================================================
# HOW TO USE LLAMA-INDEX PRE-BUILT CHUNKING STRATEGIES
# ============================================================================

"""
INTEGRATION GUIDE: Using llama-index Node Parsers
=================================================

Instead of implementing chunking strategies by hand, use llama-index's pre-built node parsers!
All parsers are available from: llama_index.core.node_parser

OPTION 1: Upgrade Current Implementation to SentenceSplitter
------------------------------------------------------------
Replace TokenTextSplitter with SentenceSplitter for better results:

```python
# In token_chunking.py, modify the TokenChunking class:

from llama_index.core.node_parser import SentenceSplitter

class SentenceChunking(ChunkingBase):  # Or rename TokenChunking
    def __init__(self):
        self.DEFAULT_CHUNK_SIZE = 1024  # Larger chunks work well with sentences
        self.DEFAULT_CHUNK_OVERLAP = 200
    
    def _text_splitter(self):
        print("Running sentence chunker...")        
        splitter = SentenceSplitter(
            chunk_size=self.DEFAULT_CHUNK_SIZE,
            chunk_overlap=self.DEFAULT_CHUNK_OVERLAP
        )
        chunks = splitter.split_text(self.text)
        return chunks
```

OPTION 2: Add Semantic Chunking (Topic-Based) - REQUIRES AZURE OPENAI EMBEDDINGS
--------------------------------------------------------------------------------
Create a new file: src/ingestion/chunking/semantic_chunking.py

This strategy uses embeddings to detect topic changes, so it needs an embedding model.
It will use the Azure OpenAI embeddings configured in your .env file.

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from src.ingestion.chunking.chunking_base import ChunkingBase
import os

class SemanticChunking(ChunkingBase):
    def __init__(self, embedding_model: str = "text-embedding-3-large"):
        '''
        Initialize semantic chunking with Azure OpenAI embeddings.
        Uses credentials from .env file (same as EmbeddingsService).
        '''
        # Configure Azure OpenAI embeddings for llama-index
        # AzureOpenAIEmbedding properly handles Azure OpenAI endpoints
        embed_model = AzureOpenAIEmbedding(
            model=os.getenv("AZURE_EMBEDDINGS_MODEL_NAME", "text-embedding-3-large"),
            deployment_name=os.getenv("AZURE_EMBEDDINGS_DEPLOYMENT_NAME", "text-embedding-3-large"),
            api_key=os.getenv("AZURE_EMBEDDINGS_API_KEY"),
            azure_endpoint=os.getenv("AZURE_EMBEDDINGS_ENDPOINT"),
            api_version=os.getenv("AZURE_EMBEDDINGS_API_VERSION", "2024-02-01"),
        )
        
        # Initialize semantic splitter with Azure embeddings
        # buffer_size=1: Compare consecutive sentences
        # breakpoint_percentile_threshold=95: Split when similarity drops below 95th percentile
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model  # Azure OpenAI embeddings from .env
        )
    
    def _text_splitter(self):
        print("Running semantic chunker with Azure OpenAI embeddings...")
        # SemanticSplitterNodeParser works with Document objects
        doc = Document(text=self.text)
        nodes = self.splitter.get_nodes_from_documents([doc])
        # Extract text from nodes
        return [node.text for node in nodes]
    
    def get_chunks_length(self):
        return len(self.chunks)
    
    def get_chunks_from_text(self, text: str) -> list:
        self.text = text
        self.chunks = self._text_splitter()
        return self.chunks
    
    def get_metadata(self, node):
        # Can access node.metadata, node.start_char_idx, node.end_char_idx
        raise NotImplementedError
```

NOTE: SemanticSplitterNodeParser will make embedding API calls for each sentence,
      so it's slower and more expensive than SentenceSplitter. Use for documents
      where topic-based chunking is critical (e.g., long research papers).


OPTION 3: Add Sentence Window Chunking (Context-Aware)
------------------------------------------------------
Create a new file: src/ingestion/chunking/window_chunking.py

```python
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document
from src.ingestion.chunking.chunking_base import ChunkingBase

class WindowChunking(ChunkingBase):
    def __init__(self):
        self.splitter = SentenceWindowNodeParser(
            window_size=3,  # 3 sentences before and after
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
    
    def _text_splitter(self):
        print("Running window chunker...")
        doc = Document(text=self.text)
        nodes = self.splitter.get_nodes_from_documents([doc])
        return [node.text for node in nodes]
    
    # ... implement other methods similar to SemanticChunking
```

OPTION 4: Strategy Selection in text_to_chunks()
------------------------------------------------
Modify the text_to_chunks() function to support multiple strategies:

```python
def text_to_chunks(text: str, strategy: str = "sentence", chunk_size: int = 1024) -> list[str]:
    '''
    Chunk text using specified strategy.
    
    Args:
        text: Input text to chunk
        strategy: One of "token", "sentence", "semantic", "window"
        chunk_size: Target chunk size (ignored for semantic/window strategies)
    
    Returns:
        list[str]: Text chunks
    '''
    if strategy == "token":
        chunker = TokenChunking()
    elif strategy == "sentence":
        # Import here to avoid circular imports
        from src.ingestion.chunking.sentence_chunking import SentenceChunking
        chunker = SentenceChunking()
    elif strategy == "semantic":
        from src.ingestion.chunking.semantic_chunking import SemanticChunking
        chunker = SemanticChunking()
    elif strategy == "window":
        from src.ingestion.chunking.window_chunking import WindowChunking
        chunker = WindowChunking()
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'token', 'sentence', 'semantic', or 'window'")
    
    return chunker.get_chunks_from_text(text)
```

WHERE TO INTEGRATE:
==================
1. Choose a strategy from above (SentenceSplitter is a good starting point)
2. Create a new class file or modify token_chunking.py
3. Update faiss_index.py to use the new strategy:
   ```python
   # In faiss_index.py, line ~170:
   from src.ingestion.chunking.token_chunking import text_to_chunks
   
   # Change to:
   chunks = text_to_chunks(text, strategy="sentence")  # or "semantic", "window"
   ```
4. Test with your documents to see which strategy gives best results

RECOMMENDED APPROACH:
====================
Start by upgrading to SentenceSplitter (Option 1) - it's a drop-in replacement
that will immediately improve chunking quality with minimal code changes!
"""