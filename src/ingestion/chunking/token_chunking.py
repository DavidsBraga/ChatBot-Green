from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import Document
from src.ingestion.chunking.chunking_base import ChunkingBase
from llama_index.core import SimpleDirectoryReader




class TokenChunking(ChunkingBase):
    def __init__(self):
        self.DEFAULT_CHUNK_SIZE=500
        self.DEFAULT_CHUNK_OVERLAP=100
    
    def _text_splitter(self):
        print("Running token chunker...")        
        splitter = TokenTextSplitter(
            chunk_size=self.DEFAULT_CHUNK_SIZE,
            chunk_overlap=self.DEFAULT_CHUNK_OVERLAP
        )
        chunks = splitter.split_text(self.text)
        return chunks
    
    def get_chunks_lenght(self):
        """Returns the number of chunks for the document."""
        return len(self.chunks)
    
    def get_chunks_from_text(self, text:str) -> list:
        self.text = text
        self.chunks= self._text_splitter()
        return self.chunks
    
    def get_metadata(self, node):
        raise NotImplementedError
 


def text_to_chunks(text: str, chunk_size:int = 512) -> list[str]:
    """Splits text into chunks of a specified size.

    Args:
        text (str): The text to split into chunks.
        chunk_size (int): The desired size of each chunk (default is 512 words).

    Returns:
        list: A list of text chunks.
    """
    chunker = TokenChunking()

    chunks = chunker.get_chunks_from_text(text)
    return chunks