"""
ChromaDB Utility Functions for Quran-Tafseer Database

Provides functions to inspect and retrieve metadata from the ChromaDB collection.
"""


import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store", "chromadb_quran_tafsir")


def get_collection_metadata(db_path: str = DEFAULT_DB_PATH) -> dict:
    """
    Returns metadata about the ChromaDB collection.
    
    Args:
        db_path: Path to the ChromaDB directory
        
    Returns:
        Dictionary containing:
        - document_count: Total number of documents
        - unique_surahs: List of unique surah names
        - unique_tafsir_books: List of unique tafsir book names
        - revelation_types: List of revelation types (Meccan/Medinan)
    """
    if not os.path.exists(db_path):
        return {"error": f"Database not found at {db_path}"}
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Load the vectorstore
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    # Get the underlying collection
    collection = vectorstore._collection
    
    # Get all metadata
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data.get("metadatas", [])
    
    # Extract unique values
    unique_surahs = set()
    unique_tafsir_books = set()
    revelation_types = set()
    
    for meta in metadatas:
        if meta:
            if "surah_name" in meta:
                unique_surahs.add(meta["surah_name"])
            if "tafsir_book" in meta:
                unique_tafsir_books.add(meta["tafsir_book"])
            if "revelation_type" in meta:
                revelation_types.add(meta["revelation_type"])
    
    return {
        "document_count": len(metadatas),
        "unique_surahs": sorted(list(unique_surahs)),
        "unique_tafsir_books": sorted(list(unique_tafsir_books)),
        "revelation_types": sorted(list(revelation_types)),
        "surah_count": len(unique_surahs),
        "tafsir_book_count": len(unique_tafsir_books)
    }


def get_sample_documents(db_path: str = DEFAULT_DB_PATH, n: int = 5) -> list:
    """
    Returns sample documents from the ChromaDB collection.
    
    Args:
        db_path: Path to the ChromaDB directory
        n: Number of sample documents to return
        
    Returns:
        List of dictionaries with document content and metadata
    """
    if not os.path.exists(db_path):
        return [{"error": f"Database not found at {db_path}"}]
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    collection = vectorstore._collection
    all_data = collection.get(include=["documents", "metadatas"], limit=n)
    
    samples = []
    for i, (doc, meta) in enumerate(zip(all_data.get("documents", []), all_data.get("metadatas", []))):
        samples.append({
            "index": i,
            "content": doc,
            "metadata": meta
        })
    
    return samples


if __name__ == "__main__":
    import json
    
    print("=" * 50)
    print("ChromaDB Quran-Tafseer Collection Metadata")
    print("=" * 50)
    
    metadata = get_collection_metadata()
    
    if "error" in metadata:
        print(f"Error: {metadata['error']}")
    else:
        print(f"\nDocument Count: {metadata['document_count']}")
        print(f"Surah Count: {metadata['surah_count']}")
        print(f"Tafsir Book Count: {metadata['tafsir_book_count']}")
        print(f"\nRevelation Types: {metadata['revelation_types']}")
        print(f"\nTafsir Books:\n{json.dumps(metadata['unique_tafsir_books'], ensure_ascii=False, indent=2)}")
        
        print("\n" + "=" * 50)
        print("Sample Documents")
        print("=" * 50)
        
        samples = get_sample_documents(n=3)
        for sample in samples:
            print(f"\n--- Sample {sample['index'] + 1} ---")
            print(f"Metadata: {sample['metadata']}")
            print(f"Content: {sample['content']}")
