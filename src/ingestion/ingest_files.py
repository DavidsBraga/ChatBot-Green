import os
from src.services.vectorial_db.faiss_index import FAISSIndex
from src.ingestion.loaders.loader import Loader


# TODO: Check if the existing files in data/ will result in duplicated chunks
DATA_FOLDER = 'data'

def ingest_files_data_folder(index: FAISSIndex):
    """Ingests all files in the data folder into the FAISS index."""
    for file in os.listdir(DATA_FOLDER):
        if os.path.isdir(os.path.join(DATA_FOLDER, file)):
            # ignore directories
            continue
        
        # Skip CSV files as they contain product data, not document content
        # CSV files should be handled separately for product recommendations
        if file.endswith('.csv'):
            print(f"Skipping {file} (CSV files are for product recommendations, not document ingestion)")
            continue
        
        loader = Loader(extension=file.split(".")[-1], filepath=os.path.join(DATA_FOLDER, file))
        print(f"Ingesting {file}")
        if file.endswith('.pdf'):
            for page_num, text in loader.loader.extract_text_by_page():
                index.ingest_text(text=text, metadata={"source": file, "page": page_num})
        else:
            text = loader.extract_text()
            index.ingest_text(text=text, metadata={"source": file})

if __name__ == "__main__":
    from src.services.models.embeddings import Embeddings
    from faiss_index import FAISSIndex
    
    # run with `python -m src.ingestion.ingest_files_data_folder`

    embeddings = Embeddings()
    index = FAISSIndex(embeddings=embeddings.get_embeddings)    
    ingest_files_data_folder(index)
    index.save_index()