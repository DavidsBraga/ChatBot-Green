from abc import ABC, abstractmethod

class LoaderBase(ABC):
    """
    Abstract base class for file loaders.

    This class defines the common interface for all file loaders. 
    Concrete loader classes (e.g., for PDF, DOCX) should inherit from this class 
    and implement the abstract methods.

    Methods:
        __init__(filepath: str): Constructor for the LoaderBase class.
        extract_metadata(): Abstract method to extract metadata from a file.
        extract_text(): Abstract method to extract text content from a file.
    """
    @abstractmethod
    def __init__(self, filepath:str):
        """
        Constructor for the LoaderBase class.

        Args:
            filepath (str): The path to the file to be loaded.
        """
        pass
    
    @abstractmethod
    def extract_metadata(self):
        """
        Abstract method to extract metadata from a file.

        Returns:
            dict: A dictionary containing the extracted metadata.
        """
        pass

    @abstractmethod
    def extract_text(self):
        """
        Abstract method to extract text content from a file.

        Returns:
            str: The extracted text from the file.
        """
        pass
