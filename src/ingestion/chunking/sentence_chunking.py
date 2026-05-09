from llama_index.core.node_parser import SentenceSplitter
from src.ingestion.chunking.chunking_base import ChunkingBase

class SentenceChunking(ChunkingBase):
    
    def __init__(self):
        self.DEFAULT_CHUNK_SIZE = 1024
        self.DEFAULT_CHUNK_OVERLAP = 200
    
    def _text_splitter(self):
        print("Running sentence chunker...")
        splitter = SentenceSplitter(
            chunk_size=self.DEFAULT_CHUNK_SIZE,
            chunk_overlap=self.DEFAULT_CHUNK_OVERLAP
        )
        chunks = splitter.split_text(self.text)
        return chunks
    
    def get_chunks_length(self):
        return len(self.chunks)
    
    def get_chunks_from_text(self, text: str) -> list:
        self.text = text
        self.chunks = self._text_splitter()
        return self.chunks
    
    def get_metadata(self, node):
        raise NotImplementedError