from src.ingestion.loaders.loaderBase import LoaderBase
import html2text

class LoaderHTML(LoaderBase):

    def __init__(self,filepath:str):
        self.filepath=filepath
    

    def extract_metadata(self):
        #Dont really know how to extract metadate from html.
        #Anyway, this value is check later in the code if its empty and handled.
        return ""
    
    def extract_text(self):

        html_content=self.get_text_from_html(self.filepath)
        
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_mailto_links = True
        markdown_content = h.handle(html_content)
        #nodes=chunker.get_nodes_from_text(markdown_content)
        
        return markdown_content
        

    def get_text_from_html(self,path:str):
        
        with open(path, "r", encoding="utf-8") as infile:
            html_text_content=infile.read()

        return html_text_content
        

        
            