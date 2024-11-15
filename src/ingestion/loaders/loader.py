from src.ingestion.loaders.loaderBase import LoaderBase
from src.ingestion.loaders.loaderDOCX import LoaderDOCX
from src.ingestion.loaders.loaderHTML import LoaderHTML
from src.ingestion.loaders.loaderPDF import LoaderPDF
from src.ingestion.loaders.loaderPPTX import LoaderPPTX

class Loader:
    """
    Factory class for creating specific file loader objects based on file extension.

    Attributes:
        extension (str): The file extension of the file to be loaded.
        filepath (str): The path to the file to be loaded.
        loader (LoaderBase): The specific loader object created based on the file extension.

    Methods:
        _get_specific_loader(): Returns a specific loader object based on the file extension.
        extract_metadata(): Extracts metadata from the file using the specific loader.
        extract_text(): Extracts text from the file using the specific loader.
    """
    def __init__(self, filepath:str , extension:str) -> None:
        """Initializes the Loader class with file information and creates a specific loader object.

        Args:
            filepath (str): The path to the file to be loaded.
            extension (str): The file extension of the file to be loaded.
        """
        self.extension=extension
        self.filepath=filepath
        self.loader=self._get_specific_loader()

    def _get_specific_loader(self) -> LoaderBase:
        """Returns a specific loader object based on the file extension.

        Returns:
            LoaderBase: A specific loader object (e.g., LoaderPDF, LoaderDOCX) based on the file extension.

        Raises:
            ValueError: If the file extension is not supported.
        """
        match self.extension:
            case "pdf":
                return LoaderPDF(self.filepath)
            case "html":
                return LoaderHTML(self.filepath)
            case "pptx":
                return LoaderPPTX(self.filepath)
            case "docx":
                return LoaderDOCX(self.filepath)
            case _:
                raise ValueError(f"Not a supported extension: {self.extension}")

    def extract_metadata(self):
        """Extracts metadata from the file using the specific loader.

        Returns:
            dict: A dictionary containing the extracted metadata.
        """
        return self.loader.extract_metadata()

    def extract_text(self):
        """Extracts text from the file using the specific loader.

        Returns:
            str: The extracted text from the file.
        """
        return self.loader.extract_text()
