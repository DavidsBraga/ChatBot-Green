"""
Factory pattern implementation for document loaders.

This module provides a unified interface for loading different file types.
The Loader class acts as a factory that instantiates the appropriate loader
based on the file extension, enabling polymorphic behavior for document processing.
"""

from src.ingestion.loaders.loaderBase import LoaderBase
from src.ingestion.loaders.loaderDOCX import LoaderDOCX
from src.ingestion.loaders.loaderHTML import LoaderHTML
from src.ingestion.loaders.loaderPDF import LoaderPDF
from src.ingestion.loaders.loaderPPTX import LoaderPPTX

# TODO: Import CSV loader when implemented
# from src.ingestion.loaders.loaderCSV import LoaderCSV


class Loader:
    """
    Factory class for creating specific file loader objects based on file extension.
    
    This class implements the Factory design pattern to dynamically instantiate
    the appropriate loader class based on the file type. It provides a unified
    interface for extracting text and metadata from various document formats.

    Attributes:
        extension (str): The file extension of the file to be loaded (e.g., 'pdf', 'docx').
        filepath (str): The absolute or relative path to the file to be loaded.
        loader (LoaderBase): The specific loader instance created based on the file extension.

    Methods:
        _get_specific_loader(): Factory method that returns the appropriate loader instance.
        extract_metadata(): Extracts metadata from the file using the specific loader.
        extract_text(): Extracts text content from the file using the specific loader.
    
    Example:
        >>> loader = Loader(filepath="document.pdf", extension="pdf")
        >>> text = loader.extract_text()
        >>> metadata = loader.extract_metadata()
    """
    
    def __init__(self, filepath: str, extension: str) -> None:
        """
        Initializes the Loader with file information and creates the specific loader.
        
        The constructor stores the file path and extension, then immediately calls
        the factory method to instantiate the appropriate loader for the file type.

        Args:
            filepath (str): The path to the file to be loaded. Can be absolute or relative.
            extension (str): The file extension (without the dot) that determines the loader type.
                           Supported values: 'pdf', 'html', 'pptx', 'docx'
        
        Raises:
            ValueError: If the extension is not supported by any available loader.
        """
        self.extension = extension
        self.filepath = filepath
        # Factory method creates the appropriate loader instance
        self.loader = self._get_specific_loader()

    def _get_specific_loader(self) -> LoaderBase:
        """
        Factory method that instantiates the appropriate loader based on file extension.
        
        This method uses Python's match-case statement (Python 3.10+) to determine
        which concrete loader class to instantiate. Each loader is specialized for
        a specific file format and implements the LoaderBase interface.

        Returns:
            LoaderBase: A concrete loader instance (LoaderPDF, LoaderDOCX, etc.) that
                       implements the LoaderBase abstract class.

        Raises:
            ValueError: If the file extension is not supported. This typically happens
                       when trying to load file types that don't have an implemented loader.
        
        Note:
            The match-case statement requires Python 3.10 or higher. If using an older
            Python version, this should be replaced with if-elif-else statements.
        """
        match self.extension:
            case "pdf":
                # PDF files - uses PyPDF2 library
                return LoaderPDF(self.filepath)
            
            case "html":
                # HTML files - uses html2text library for conversion
                return LoaderHTML(self.filepath)
            
            case "pptx":
                # PowerPoint files - uses python-pptx library
                return LoaderPPTX(self.filepath)
            
            case "docx":
                # Word documents - uses python-docx library
                return LoaderDOCX(self.filepath)
            
            # TODO: Implement CSV loader for sustainable products data
            # The bootcamp challenge requires loading product recommendations from CSV files.
            # case "csv":
            #     return LoaderCSV(self.filepath)
            
            # TODO: Consider adding support for additional formats, if needed:
            # - xlsx (Excel files) - could use openpyxl or pandas
            # - txt (Plain text files) - simple file reading
            # - md (Markdown files) - could use markdown library
            # - json (JSON files) - could use json library for structured data
            # - jpg/png (Image files) - could use OCR for text extraction from images or summarization of an image content
            
            case _:
                # Default case: unsupported file extension
                raise ValueError(f"Not a supported extension: {self.extension}")

    def extract_metadata(self):
        """
        Extracts metadata from the file using the instantiated specific loader.
        
        Metadata typically includes information such as:
        - Author
        - Title
        - Creation/modification dates
        - Subject
        - Keywords
        
        The specific metadata fields available depend on the file format and
        what the specific loader implementation supports.

        Returns:
            dict: A dictionary containing the extracted metadata. The exact keys
                 and values depend on the file type and the specific loader implementation.
                 Returns False if metadata extraction fails or if all values are empty.
        
        Example:
            >>> loader = Loader(filepath="document.pdf", extension="pdf")
            >>> metadata = loader.extract_metadata()
            >>> print(metadata['author'])
        """
        return self.loader.extract_metadata()

    def extract_text(self):
        """
        Extracts the main text content from the file using the specific loader.
        
        This method delegates to the specific loader's extract_text() implementation.
        The extraction process varies by file type:
        - PDF: Extracts text from all pages
        - DOCX: Extracts text from paragraphs, tables, etc.
        - HTML: Converts HTML to plain text
        - PPTX: Extracts text from slides and notes
        
        The returned text is typically cleaned and formatted for downstream processing
        such as chunking and embedding generation.

        Returns:
            str: The complete text content extracted from the file. This is a single
                string that may contain newlines and other formatting characters.
        
        Example:
            >>> loader = Loader(filepath="climate_report.pdf", extension="pdf")
            >>> text = loader.extract_text()
            >>> print(f"Extracted {len(text)} characters")
        
        Note:
            The quality of text extraction depends on the file format and content.
            Scanned PDFs (images) will not be extracted unless OCR is implemented.
        """
        return self.loader.extract_text()
