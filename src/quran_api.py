################################################################################
# quran_api.py - FastAPI Server for Quran RAG Agent
################################################################################

"""
FastAPI application for the Quran RAG Agent.
Provides endpoints for chat interactions and vector database management.

To run:
    uvicorn quran_api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure environment variables are loaded
from dotenv import load_dotenv
load_dotenv()

# Import core logic from existing scripts
# We'll use chromadb_utils for DB management
import chromadb_utils

# We need the agent logic. To avoid side-effects from importing the script directly 
# (which might run initialization code), we should ideally refactor quran_rag_agent.py.
# However, for now, we will import what we need assuming the script is import-safe 
# or we'll replicate the necessary parts here to be safer.
# Given quran_rag_agent.py has global execution, extracting the logic is safer.

import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import quran_rag_agent
from quran_rag_agent import process_query

from prompts import AGENT1_SYSTEM_PROMPT, AGENT2_SYSTEM_PROMPT

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION (Replicated from quran_rag_agent.py) ---
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store", "chromadb_quran_tafsir")
HF_DATASET = "MohamedRashad/Quran-Tafseer"

# --- GLOBAL STATE ---
# Removed to rely completely on `quran_rag_agent.py` single source of truth

# --- MODELS ---

class Message(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    messages: List[Message]
    k: int = Field(default=3, description="عدد التفاسير لكل آية")
    
class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    response: str
    sources: List[SourceDocument] = []

class MetadataResponse(BaseModel):
    document_count: int
    unique_surahs: List[str]
    unique_tafsir_books: List[str]
    revelation_types: List[str]
    surah_count: int
    tafsir_book_count: int

# --- INITIALIZATION ---
# Handled automatically when quran_rag_agent is imported.

# --- FASTAPI APP ---
app = FastAPI(
    title="Quran Agent API",
    description="API for Quranic Agent and Vector Database Management",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup Event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up... Connecting via quran_rag_agent!")
    if quran_rag_agent.vectorstore is None:
        logger.warning("Vectorstore is not loaded properly in quran_rag_agent!")

# --- UTILS ---

# Removed unused parse_sources_from_messages as we directly return sources now

# --- ENDPOINTS ---

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Quran RAG API is running"}

@app.post("/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Chat with the RAG agent.
    """
    if not quran_rag_agent.llm_agent1 or not quran_rag_agent.llm_agent2:
        raise HTTPException(status_code=503, detail="Agents not initialized in core module")

    # The user message is the last message
    user_message = request.messages[-1].content
    k_value = request.k

    # Run agent in thread
    try:
        def run_agents():
            return process_query(user_message, k_value)

        response_text, chunks_data = await asyncio.to_thread(run_agents)
        
        # Sources formatting
        sources = []
        for chunk in chunks_data:
            sources.append(SourceDocument(
                content=chunk["content"], 
                metadata={"type": "retrieved_chunk", **chunk["metadata"]}
            ))

        return ChatResponse(
            response=response_text,
            sources=sources
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/db/metadata", response_model=MetadataResponse, tags=["Database"])
async def get_metadata():
    """
    Get metadata about the vector database (counts, unique books, etc.).
    """
    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=404, detail="Database not found")
        
    try:
        # Run blocking DB call in thread
        metadata = await asyncio.to_thread(chromadb_utils.get_collection_metadata, CHROMA_PATH)
        if "error" in metadata:
             raise HTTPException(status_code=500, detail=metadata["error"])
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/db/samples", tags=["Database"])
async def get_samples(n: int = 5):
    """
    Get sample documents from the database.
    """
    try:
        samples = await asyncio.to_thread(chromadb_utils.get_sample_documents, CHROMA_PATH, n)
        return samples
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/db/search", tags=["Database"])
async def search_db(query: str = Body(..., embed=True), k: int = 5):
    """
    Direct semantic search against the vector database (bypassing the agent).
    Useful for debugging or simple retrieval.
    """
    if not quran_rag_agent.vectorstore:
         raise HTTPException(status_code=503, detail="Vectorstore not loaded")
         
    try:
        # We need to run this in a thread because similarity_search might be blocking (depending on impl)
        # Chroma's similarity_search is often synchronous.
        results = await asyncio.to_thread(quran_rag_agent.vectorstore.similarity_search, query, k=k)
        
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/db/chunk/{dataset_index}", tags=["Database"])
async def get_chunk_by_index(dataset_index: int):
    """
    Get a specific chunk by its dataset index.
    """
    if not quran_rag_agent.vectorstore:
         raise HTTPException(status_code=503, detail="Vectorstore not loaded")

    try:
        # User requested "index from vector space".
        # Since we cannot re-ingest to add metadata, we use the 'offset' parameter
        # which retrieves the N-th item in the database.
        results = await asyncio.to_thread(
            quran_rag_agent.vectorstore.get,
            limit=1,
            offset=dataset_index
        )
        
        if not results['ids']:
            raise HTTPException(status_code=404, detail=f"Chunk with index {dataset_index} not found")

        # Chroma's get returns dicts of lists
        doc = {
            "content": results['documents'][0],
            "metadata": results['metadatas'][0],
            "id": results['ids'][0]
        }
        return doc
    except HTTPException:
        raise
    except Exception as e:
        # If dataset_index is not found or other error
        logger.error(f"Error retrieving chunk {dataset_index}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/db/reload", tags=["Database"])
async def reload_db():
    """
    Force reload of the vector store (e.g. after re-ingestion).
    """
    try:
        await asyncio.to_thread(quran_rag_agent.load_data)
        return {"status": "success", "message": "Agent and Database reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
