"""
FAISS Vector Database Index Manager

This module manages the FAISS (Facebook AI Similarity Search) vector index used for
semantic search in the RAG (Retrieval-Augmented Generation) architecture. It handles:
- Creating and maintaining the vector index
- Embedding and storing text chunks
- Performing similarity search for query retrieval
- Persisting and loading the index from disk

FAISS is a library for efficient similarity search and clustering of dense vectors.
It uses L2 (Euclidean) distance for finding the nearest neighbors to a query vector.
"""

from faiss import IndexFlatL2, write_index, read_index
import numpy as np
import os

from src.ingestion.chunking.token_chunking import text_to_chunks


class FAISSIndex():
    """
    Manages a FAISS index for storing and retrieving text chunks based on their embeddings.
    
    This class is a core component of the RAG system, providing the "Retrieval" functionality.
    It stores document chunks as high-dimensional vectors (embeddings) and enables fast
    semantic search to find the most relevant chunks for a given query.
    
    The workflow is:
    1. Text documents are chunked into smaller pieces
    2. Each chunk is converted to an embedding vector using OpenAI's embedding model
    3. Vectors are stored in the FAISS index with their corresponding text
    4. When a query arrives, it's converted to a vector and compared against stored vectors
    5. The most similar chunks are retrieved to provide context to the LLM

    Attributes:
        dimension (int): The dimensionality of the embedding vectors. Must match the
                        embedding model's output dimension (e.g., 3072 for text-embedding-3-large).
        embeddings (function): A callable that takes text as input and returns an embedding
                              vector. Typically points to Embeddings.get_embeddings method.
        index (faiss.IndexFlatL2): The FAISS index object that stores vectors and performs
                                   similarity search using L2 (Euclidean) distance.
        chunks_list (list): Parallel array storing the actual text chunks. The index i in
                           this list corresponds to the vector at index i in the FAISS index.

    Methods:
        _create_faiss_index(): Initializes a new empty FAISS index.
        ingest_text(): Adds text chunks to the index after converting them to embeddings.
        retrieve_chunks(): Performs semantic search to find relevant chunks for a query.
        save_index(): Persists the index and chunks to disk for later use.
        load_index(): Loads a previously saved index and chunks from disk.
    
    Example:
        >>> embeddings = Embeddings()
        >>> index = FAISSIndex(embeddings=embeddings.get_embeddings, dimension=3072)
        >>> index.ingest_text("Climate change is a serious issue.")
        >>> results = index.retrieve_chunks("What is climate change?", num_chunks=3)
    
    TODO: Add metadata tracking for source attribution (which document each chunk came from)
    """
    
    def __init__(self, dimension: int = 1536, embeddings=None):
        """
        Initializes the FAISS index manager with embedding configuration.
        
        Creates a new FAISS index with the specified dimension. The dimension must match
        the output size of the embedding model being used:
        - text-embedding-ada-002: 1536 dimensions
        - text-embedding-3-small: 1536 dimensions  
        - text-embedding-3-large: 3072 dimensions (current default in .env)

        Args:
            dimension (int): The dimension of the embeddings. Defaults to 1536 but should
                           be set to match your embedding model (3072 for text-embedding-3-large).
            embeddings (function): A callable that generates embeddings from text.
                                 Should accept a string and return a list of floats.
                                 Typically set to Embeddings().get_embeddings

        Raises:
            ValueError: If no embeddings function is provided.
        
        Note:
            The embeddings function is called for every chunk during ingestion and
            for every query during retrieval, so it should be efficiently implemented.
        """
        if not embeddings:
            raise ValueError("No embeddings provided.")
        
        # Store the embedding function (dependency injection pattern)
        self.embeddings = embeddings
        
        # Store the vector dimension
        self.dimension = dimension
        
        # FAISS index will be created by the helper method
        self.index: IndexFlatL2 | None = None
        self._create_faiss_index()
        
        # Parallel list to store the actual text chunks
        # chunks_list[i] corresponds to the vector at index i in the FAISS index
        self.chunks_list: list = []
    
    def _create_faiss_index(self):
        """
        Initializes a new FAISS IndexFlatL2 for exact similarity search.
        
        IndexFlatL2 performs exact (brute-force) L2 distance search, which means it
        compares the query vector against every vector in the index. This is:
        - Pros: Exact results, no approximation errors
        - Cons: Slower for very large datasets (>1M vectors)
        
        For the bootcamp challenge with a few documents, this is perfectly adequate.
        For production systems with millions of documents, consider using approximate
        indexes like IndexIVFFlat or IndexHNSWFlat.
        
        The index is stored in self.index and is ready to receive vectors via add().
        """
        self.index = IndexFlatL2(self.dimension)
    
    def ingest_text(self, text: str | None = None, text_chunks: list | None = None) -> bool:
        """
        Ingests text into the FAISS index by converting chunks to embeddings and storing them.
        
        This method is the main entry point for adding documents to the vector database.
        It supports two modes:
        1. Provide raw text - will be automatically chunked using the token chunker
        2. Provide pre-chunked text - will use chunks as-is
        
        The process for each chunk:
        1. Generate embedding vector using the configured embeddings function
        2. Convert to float32 numpy array (required by FAISS)
        3. Add vector to FAISS index
        4. Store the original text chunk in parallel chunks_list
        
        Args:
            text (str | None): Raw text to be chunked and ingested. If provided,
                              text_chunks should be None.
            text_chunks (list | None): Pre-chunked text to be ingested directly.
                                      If provided, text should be None.
        
        Returns:
            bool: Always returns True if ingestion succeeds.
        
        Raises:
            ValueError: If neither text nor text_chunks is provided.
        
        TODO: Improve chunking strategy - current implementation uses simple token-based chunking
              Consider:
              - Semantic chunking (split at sentence/paragraph boundaries)
              - Recursive chunking for better context preservation
              - Sliding window with overlap for continuity
              - Metadata-aware chunking (preserve document structure)
        
        TODO: Add source tracking - store which document each chunk came from
              This is needed for source citation in responses        
        Note:
            Each embedding API call has a cost, so chunking strategy significantly
            impacts both cost and retrieval quality.
        """
        # Validate that we have input to work with
        if not (text_chunks or text):
            raise ValueError("Either text or text_chunks must be provided")
        
        # If raw text provided, chunk it using the token chunker
        if not text_chunks:
            # TODO: Improve chunking strategy (see TODO above)
            # Current implementation uses TokenTextSplitter from llama-index
            # with fixed chunk_size=500 and chunk_overlap=100
            text_chunks = text_to_chunks(text)
        
        # Process each chunk: embed and store
        for chunk in text_chunks:
            # Generate embedding vector for this chunk (API call to OpenAI)
            embedding = self.embeddings(chunk)
            
            # Convert to numpy array with float32 dtype (FAISS requirement)
            # Shape is (1, dimension) because FAISS add() expects a 2D array
            embedding_array = np.array([embedding]).astype('float32')
            
            # Add the vector to the FAISS index
            # The index automatically assigns an integer ID (0, 1, 2, ...)
            self.index.add(embedding_array)
            
            # Store the original text chunk at the same index
            # This maintains the correspondence: chunks_list[i] ↔ FAISS vector[i]
            self.chunks_list += [chunk]
        
        return True
    
    def retrieve_chunks(self, query: str, num_chunks: int = 5) -> list:
        """
        Retrieves the most relevant text chunks for a given query using semantic search.
        
        This is the core retrieval function in the RAG pipeline. It performs the "R" in RAG
        by finding the most semantically similar chunks to the user's query.
        
        The process:
        1. Convert the query text to an embedding vector (same model as used for chunks)
        2. Perform nearest neighbor search in the FAISS index using L2 distance
        3. Return the text of the k-nearest chunks
        
        The returned chunks will be used as context for the LLM to generate answers.
        
        Args:
            query (str): The user's question or search query.
            num_chunks (int): The number of most relevant chunks to retrieve.
                            Defaults to 5. More chunks = more context but longer prompts.
        
        Returns:
            list: A list of text chunks (strings) ranked by relevance to the query.
                 The first chunk is the most relevant, last is least relevant.
        
        Example:
            >>> index.retrieve_chunks("What causes climate change?", num_chunks=3)
            ['Greenhouse gases trap heat...', 'CO2 emissions from...', 'Deforestation contributes...']
        
        TODO: Add source citation - return (chunk, source_document, page_number) tuples
              instead of just chunks. This would enable the chatbot to cite sources.
        """
        # Generate embedding for the query using the same embedding model
        query_embedding = self.embeddings(query)
        
        # Convert to numpy array with float32 dtype for FAISS
        # Shape is (1, dimension) for a single query vector
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search the FAISS index for the k nearest neighbors
        # Returns: D = distances (lower is better), I = indices of matched vectors
        # We use _ to ignore distances since we only need the indices
        _, I = self.index.search(query_vector, num_chunks)
        
        # Retrieve the actual text chunks using the indices
        # I[0] because search returns a 2D array (supports batch queries)
        return [self.chunks_list[i] for i in I[0]]
    
    def save_index(self, path=r"./faiss_index"):
        """
        Persists the FAISS index and chunks to disk for later use.
        
        This method saves two files:
        1. index.faiss - Binary file containing the FAISS index structure and all vectors
        2. chunks.npy - NumPy binary file containing the parallel array of text chunks
        
        Saving the index is crucial for:
        - Avoiding re-ingestion and re-embedding on every run (saves time and API costs)
        - Deploying the chatbot with a pre-built knowledge base
        - Version control of the knowledge base
        
        The index can be loaded later using load_index() to restore the complete
        vector database state.

        Args:
            path (str, optional): Directory path where the index files will be saved.
                                Defaults to "./faiss_index" in the project root.
                                The directory will be created if it doesn't exist.
        
        Side Effects:
            - Creates the specified directory if it doesn't exist
            - Writes two files: index.faiss and chunks.npy
            - Overwrites existing files if present
        
        Example:
            >>> index.ingest_text("Some document text...")
            >>> index.save_index()  # Saves to ./faiss_index/
            >>> index.save_index("./my_custom_index")  # Custom location
        
        Note:
            The saved index is tied to the embedding model used. Loading an index
            and querying with a different embedding model will give incorrect results.
        """
        print(f"Saving index to '{path}' folder...")
        
        # Construct full file paths
        index_path = os.path.join(path, "index.faiss")
        chunks_path = os.path.join(path, "chunks.npy")
        
        # Create directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save the FAISS index structure (vectors and index metadata)
        write_index(self.index, index_path)
        
        # Save the parallel chunks list as a NumPy array
        # allow_pickle=True is needed for Python object serialization
        np.save(chunks_path, self.chunks_list)
    
    def load_index(self, path: str = r"./faiss_index"):
        """
        Loads a previously saved FAISS index and chunks from disk.
        
        This method restores the complete vector database state from disk, including:
        - The FAISS index structure with all vectors
        - The parallel array of text chunks
        
        Loading a saved index is much faster than re-ingesting and re-embedding all
        documents, and avoids API costs for regenerating embeddings.

        Args:
            path (str, optional): Directory path where the index files are located.
                                Defaults to "./faiss_index".

        Raises:
            FileNotFoundError: If the specified directory doesn't exist or if required
                             files (index.faiss or chunks.npy) are missing.
        
        Side Effects:
            - Replaces the current index and chunks_list with loaded data
            - Any unsaved data in the current index will be lost
        
        Example:
            >>> index = FAISSIndex(embeddings=embeddings.get_embeddings, dimension=3072)
            >>> index.load_index()  # Loads from ./faiss_index/
            >>> results = index.retrieve_chunks("climate change", num_chunks=5)
        
        Warning:
            Loading an index created with a different embedding model will result in
            incorrect retrieval results. Always use the same model for indexing and querying.
        """
        print(f"Loading index from '{path}' folder...")
        
        # Construct full file paths
        index_path = os.path.join(path, "index.faiss")
        chunks_path = os.path.join(path, "chunks.npy")
        
        # Check if the index directory exists
        if not os.path.exists(path):
            raise FileNotFoundError("Index not found.")
        
        # Load the FAISS index from disk
        self.index = read_index(index_path)
        
        # Load the chunks array from disk
        # allow_pickle=True is needed to load Python objects
        self.chunks_list = np.load(chunks_path, allow_pickle=True).tolist()
