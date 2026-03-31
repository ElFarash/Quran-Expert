import os
import tiktoken
from datasets import load_dataset
import pandas as pd
import concurrent.futures
import time

# Configuration matching quran_rag_agent.py
HF_DATASET = "MohamedRashad/Quran-Tafseer"
EMBEDDING_MODEL_PRICE_PER_1M = 0.02  # text-embedding-3-small
ENCODING_NAME = "cl100k_base"

def count_tokens_in_chunk(chunk_df):
    """Process a chunk of the dataframe."""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    local_count = 0
    for _, row in chunk_df.iterrows():
        content = (
            f"الآية: {row['ayah']}\n"
            f"التفسير ({row['tafsir_book']}): {row['tafsir_content']}"
        )
        local_count += len(encoding.encode(content))
    return local_count

def calculate_cost_parallel():
    print("🚀 Loading dataset...")
    ds = load_dataset(HF_DATASET)
    df = ds["train"].to_pandas()
    
    total_records = len(df)
    print(f"📊 Dataset loaded: {total_records} records")
    
    print("⚡ Counting tokens using parallel processing...")
    start_time = time.time()
    
    # Determine number of workers
    num_workers = os.cpu_count() or 4
    chunk_size = total_records // num_workers + 1
    
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, total_records, chunk_size)]
    
    total_tokens = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(count_tokens_in_chunk, chunk) for chunk in chunks]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            total_tokens += result
            print(f"  Worker {i+1}/{num_workers} finished ({result:,} tokens)")
            
    elapsed = time.time() - start_time
    cost = (total_tokens / 1_000_000) * EMBEDDING_MODEL_PRICE_PER_1M
    
    print("\n" + "="*40)
    print("💰 ESTIMATED TOKEN COST REPORT")
    print("="*40)
    print(f"Total Documents: {total_records}")
    print(f"Total Tokens:    {total_tokens:,}")
    print(f"Time Taken:      {elapsed:.2f} seconds")
    print(f"Model:           text-embedding-3-small")
    print(f"Price per 1M:    ${EMBEDDING_MODEL_PRICE_PER_1M}")
    print(f"----------------------------------------")
    print(f"TOTAL ESTIMATED COST: ${cost:.4f}")
    print("="*40)

if __name__ == "__main__":
    calculate_cost_parallel()
