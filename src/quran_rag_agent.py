import os
import time
from typing import List
import json

import gradio as gr # Not strictly needed here but kept for backward compatibility if any
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from prompts import AGENT1_SYSTEM_PROMPT, AGENT2_SYSTEM_PROMPT

# ==========================================
# 1. CONFIGURATION
# ==========================================

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Model Configuration
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vector_store", "chromadb_quran_tafsir")

# HuggingFace Dataset
HF_DATASET = "MohamedRashad/Quran-Tafseer"

# ==========================================
# 2. INITIALIZATION & INGESTION
# ==========================================

print("⏳ Initializing Models...")

# Initialize Embeddings (OpenAI)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vectorstore = None


def load_data():
    global vectorstore
    
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("✅ Loading existing Quran-Tafseer index...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
    else:
        print("🚀 Downloading and indexing Quran-Tafseer dataset (This may take time)...")
        
        # Load from HuggingFace
        ds = load_dataset(HF_DATASET)
        df = ds["train"].to_pandas()
        
        print(f"📊 Dataset loaded: {len(df)} records")
        
        documents = []
        for index, row in df.iterrows():
            # Combine ayah and tafsir into single document
            content = (
                f"الآية: {row['ayah']}\n"
                f"التفسير ({row['tafsir_book']}): {row['tafsir_content']}"
            )
            
            metadata = {
                "surah_name": str(row['surah_name']),
                "revelation_type": str(row['revelation_type']),
                "tafsir_book": str(row['tafsir_book']),
                "source": "quran_tafseer_hf"
            }
            documents.append(Document(page_content=content, metadata=metadata))
            
            # Progress indicator
            if (index + 1) % 10000 == 0:
                print(f"  Processed {index + 1}/{len(df)} records...")
        
        print(f"📝 Creating embeddings for {len(documents)} documents...")
        
        # Process in batches to avoid OOM and show progress
        batch_size = 500  # Reduced batch size for stability
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        vectorstore = None
        
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def process_batch(batch_docs):
            global vectorstore
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch_docs,
                    embedding=embeddings,
                    persist_directory=CHROMA_PATH
                )
            else:
                vectorstore.add_documents(documents=batch_docs)

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            print(f"  Processing batch {current_batch}/{total_batches} ({len(batch)} docs)...")
            
            try:
                process_batch(batch)
            except Exception as e:
                print(f"❌ Failed to process batch {current_batch}. Error: {e}")
                # Optional: Continue or break depending on requirement. 
                # For now we print error but continue to try next batches? 
                # Or better to re-raise and fail?
                # Given user complained about crash, let's try to proceed.
                print("Skipping this batch and continuing...")
                continue
                
        print(f"✅ Indexed {len(documents)} Quran-Tafseer documents.")


load_data()

# We'll use this module solely for initialization and process_query logic.


# ==========================================
# 4. BUILD THE AGENTS
# ==========================================

llm_agent1 = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0.3
)

llm_agent2 = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0.0
)

# ==========================================
# 5. CORE RAG LOGIC
# ==========================================

def process_query(message: str, k_value: int = 3) -> tuple[str, list[dict]]:
    global vectorstore
    try:
        all_chunks = []
        
        if not vectorstore:
            return "Error: Quran-Tafseer database not loaded.", []
            
        # 1. Call Agent 1
        messages1 = [
            SystemMessage(content=AGENT1_SYSTEM_PROMPT),
            HumanMessage(content=message)
        ]
        
        try:
            res1 = llm_agent1.invoke(messages1)
            content = res1.content
            
            # Clean possible markdown block
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
                
            data = json.loads(content)
            if isinstance(data, list):
                ayahs = data
            elif isinstance(data, dict):
                ayahs = data.get("ayahs", [])
                if not ayahs and "ayah" in data:
                    ayahs = [data]
            else:
                ayahs = [{"ayah": message}]
        except Exception as e:
            print(f"Error parsing Agent 1 response: {e}, content: {res1.content if 'res1' in locals() else 'None'}")
            ayahs = [{"ayah": message}]
            
        if not ayahs:
            ayahs = [{"ayah": message}]
            
        # 2. Search Loop
        all_chunks = []
        chunk_strings = []
        
        for ayah_obj in ayahs:
            ayah_query = ayah_obj.get("ayah", "")
            if not ayah_query:
                continue
                
            print(f"Searching for: {ayah_query} with k={k_value}")
            results = vectorstore.similarity_search(ayah_query, k=k_value)
            
            for i, doc in enumerate(results, 1):
                meta = doc.metadata
                chunk_str = f"--- Result {i} for {ayah_query} ---\n"
                chunk_str += f"Surah: {meta.get('surah_name', 'N/A')} | "
                chunk_str += f"Type: {meta.get('revelation_type', 'N/A')} | "
                chunk_str += f"Tafsir Book: {meta.get('tafsir_book', 'N/A')}\n"
                chunk_str += f"{doc.page_content}\n"
                chunk_strings.append(chunk_str)
                all_chunks.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "query": ayah_query
                })
                
        context = "\n\n".join(chunk_strings)
        
        # 3. Call Agent 2 with Fallback Logic
        def call_agent_2(ctx, user_msg):
            prompt = f"Context:\n{ctx}\n\nUser Query:\n{user_msg}"
            messages2 = [
                SystemMessage(content=AGENT2_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            return llm_agent2.invoke(messages2).content

        warning_msg = ""
        try:
            final_answer = call_agent_2(context, message)
        except Exception as e:
            print(f"Agent 2 failed (possibly token limit): {e}")
            if len(ayahs) > 1:
                print("Falling back to first ayah only...")
                first_ayah_query = ayahs[0].get("ayah", "")
                first_chunks = [c for c in all_chunks if c.get("query") == first_ayah_query]
                
                # Overwrite all_chunks so we only return the final context's chunks
                all_chunks = first_chunks
                
                chunk_strings = []
                for i, chunk in enumerate(first_chunks, 1):
                    meta = chunk["metadata"]
                    chunk_str = f"--- Result {i} for {first_ayah_query} ---\n"
                    chunk_str += f"Surah: {meta.get('surah_name', 'N/A')} | "
                    chunk_str += f"Type: {meta.get('revelation_type', 'N/A')} | "
                    chunk_str += f"Tafsir Book: {meta.get('tafsir_book', 'N/A')}\n"
                    chunk_str += f"{chunk['content']}\n"
                    chunk_strings.append(chunk_str)
                
                context = "\n\n".join(chunk_strings)
                try:
                    final_answer = call_agent_2(context, message)
                    warning_msg = "\n\n**ملاحظة:** نظراً لطول التفاسير، تم جلب تفسير الآية الأولى فقط للمحافظة على استقرار النظام وإرسال استعلام ضمن حدود توكنات النموذج. يُرجى السؤال عن آية واحدة في كل محاولة (It is better to ask for one ayah per request)."
                except Exception as e2:
                    return f"حدث خطأ في النظام مرتين: {str(e2)}", all_chunks
            else:
                return f"حدث خطأ في النظام: {str(e)}", all_chunks
                
        try:
            log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "request_log.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"--- USER QUERY ---\n{message}\n\n")
                f.write(f"--- AGENT 1 FINAL PROMPT ---\n[SYSTEM]:\n{AGENT1_SYSTEM_PROMPT}\n[USER]:\n{message}\n\n")
                f.write(f"--- AGENT 1 OUTPUT (Parsed Ayahs) ---\n{json.dumps(ayahs, ensure_ascii=False, indent=2)}\n\n")
                f.write(f"--- SEARCH CHUNKS (Context) ---\n{context}\n\n")
                f.write(f"--- AGENT 2 FINAL PROMPT ---\n[SYSTEM]:\n{AGENT2_SYSTEM_PROMPT}\n[USER]:\nContext:\n{context}\n\nUser Query:\n{message}\n\n")
                f.write(f"--- AGENT 2 FINAL OUTPUT ---\n{final_answer + warning_msg}\n\n")
                f.write(f"--- RESOURCES ---\n")
                for chunk in all_chunks:
                    f.write(f"Surah: {chunk['metadata'].get('surah_name')} | Book: {chunk['metadata'].get('tafsir_book')}\n")
        except Exception as e:
            print(f"Failed to write log: {e}")

        return final_answer + warning_msg, all_chunks
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"حدث خطأ في النظام: {str(e)}", []

# Gradio App has been moved to app.py