from abc import ABC, abstractmethod

class ChunkingBase(ABC):

    @abstractmethod
    def __init__(self, embedding_model:str | None):
        pass

    @abstractmethod
    def _text_splitter(self):
        pass
    
    @abstractmethod
    def get_chunks_lenght(self):
        pass
    
    @abstractmethod
    def get_chunks_from_text(self,text:str):
        pass

    @abstractmethod
    def get_metadata(self, node):
        pass