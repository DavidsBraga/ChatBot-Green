from faiss import IndexFlatL2, write_index, read_index
import numpy as np
import os

from src.ingestion.chunking.token_chunking import text_to_chunks


class FAISSIndex():
    """
    Manages a FAISS index for storing and retrieving text chunks based on their embeddings.

    Attributes:
        dimension (int): The dimension of the embeddings.
        embeddings (function): The function used to generate embeddings for text.
        index (faiss.IndexFlatL2): The FAISS index object.
        chunks_list (list): A list of text chunks stored in the index.

    Methods:
        _create_faiss_index(): Initializes a new FAISS index.
        ingest_text(): Adds text chunks to the index.
        retrieve_chunks(): Retrieves relevant chunks for a given query.
        save_index(): Saves the index and chunk list to disk.
        load_index(): Loads the index and chunk list from disk.
    """
    def __init__(self, dimension: int = 1536, embeddings = None):
        """dimension(int): the dimension of the embeddings.
        embeddings(function): the function that returns the embeddings."""
        if not embeddings:
            raise ValueError("No embeddings provided.")
        self.embeddings = embeddings
        self.dimension = dimension
        self.index: IndexFlatL2 | None= None
        self._create_faiss_index()
        self.chunks_list: list = []
    
    def _create_faiss_index(self):
        self.index = IndexFlatL2(self.dimension)
    
    def ingest_text(self, text: str | None = None, text_chunks: list | None = None) -> bool:
        """Ingests text to the faiss index."""
        if not (text_chunks or text):
            raise ValueError("Either text or text_chunks must be provided")
        if not text_chunks:
            #TODO: Improve chunking
            text_chunks = text_to_chunks(text)
        for chunk in text_chunks:
            embedding = self.embeddings(chunk)
            self.index.add(np.array([embedding]).astype('float32'))
            self.chunks_list += [chunk]
        return True
    
    def retrieve_chunks(self, query: str, num_chunks: int = 5) -> list:
        """Retrieves chunks from the FAISS index based on a query."""
        query_embedding = self.embeddings(query)
        query_vector = np.array([query_embedding]).astype('float32')
        _, I = self.index.search(query_vector, num_chunks)
        return [self.chunks_list[i] for i in I[0]]
    
    def save_index(self, path=r"./faiss_index"):
        """Saves the index and chunk list to disk.

        Args:
            path (str, optional): The directory to save the index to. Defaults to r"./faiss_index".
        """
        print(f"Saving index to '{path}' folder...")
        index_path = os.path.join(path, "index.faiss")
        chunks_path = os.path.join(path, "chunks.npy")
        if not os.path.exists(path):
            os.makedirs(path)
        write_index(self.index, index_path)
        np.save(chunks_path, self.chunks_list)
    
    def load_index(self, path: str = r"./faiss_index"):
        """Loads the index and chunk list from disk.

        Args:
            path (str, optional): The directory to load the index from. Defaults to r"./faiss_index".

        Raises:
            FileNotFoundError: If the index is not found at the specified path.
        """
        print(f"Loading index from '{path}' folder...")
        index_path = os.path.join(path, "index.faiss")
        chunks_path = os.path.join(path, "chunks.npy")
        if not os.path.exists(path):
            raise FileNotFoundError("Index not found.")
        self.index = read_index(index_path)
        self.chunks_list = np.load(chunks_path, allow_pickle=True).tolist()
