from src.ingestion.loaders.loaderBase import LoaderBase
from pathlib import Path
import PyPDF2  
import io
import os
import tempfile

class LoaderPDF(LoaderBase):

    def __init__(self, filepath:str):
        self.filepath=filepath

    def extract_metadata(self):
        pdf_file_reader = PyPDF2.PdfReader(self.filepath)
        doc_info = pdf_file_reader.metadata
        metadata = {  
            'author': doc_info.author,  
            'creator': doc_info.creator,  
            'producer': doc_info.producer,  
            'subject': doc_info.subject,  
            'title': doc_info.title,  
            #'number_of_pages': len(doc_info.pages)  
        }

        self.metadata=metadata
        
        return self.metadata if self.all_keys_have_values(metadata=self.metadata) else False
    
    def extract_text(self):
        with open(self.filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += " " + reader.pages[page_num].extract_text()
        return text

    def all_keys_have_values(self, metadata, value_check=lambda x: x is not None and x != ''):
        return all(value_check(value) for value in metadata.values())
