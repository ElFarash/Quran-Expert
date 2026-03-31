# Quran RAG Agent System
A comprehensive AI-powered system designed to provide deep, scholarly Tafsir (interpretations) of the Quran. Built using LangChain, OpenAI, and ChromaDB, the system employs a robust **Dual-Agent Architecture** to accurately extract Quranic references and contextualize interpretations from a diverse pool of authenticated scholarly works.

## Features
- **Dual-Agent Workflow:**
  - **Contextualizing Agent (Quran Expert):** Precision-parses user queries to identify and extract exact Ayah (verse) references.
  - **RAG Agent (Tafsir Specialist):** Analyzes the localized embeddings and generates high-level scholarly responses grounded explicitly in the retrieved Tafsir chunks.
- **Robust Fallback Logic:** Identifies potential model token-limit exceptions when evaluating multiple scattered verses, safely downgrading to single-verse analysis and alerting the user.
- **Interactive UI:** A full Gradio-based web interface (`src/app.py`).
- **RESTful API:** A FastAPI backend (`src/quran_api.py`) for integration into external apps.


## Installation & Setup

All execution and dependency management must happen inside the `quran_llm` Conda environment.

1. **Create and activate the Conda Environment:**
```bash
conda create -n quran_llm python=3.10 -y
conda activate quran_llm
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Database Setup (Vector Space):**
To avoid generating costly and time-consuming embeddings from scratch, download the pre-computed vector space:
- Download the `chromadb_quran_tafsir` folder from this [Google Drive Link](https://drive.google.com/drive/folders/1AV9cc7-sJhM8s8ntec_nrPG2XCpi3DnL?usp=sharing).
- Place the downloaded folder in the `vector_store` directory of this project so the path becomes `./vector_store/chromadb_quran_tafsir/`.

4. **Environment Variables:**
Create a `.env` file in the project's root directory and add your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Project

### Option 1: Gradio Web Interface
To interact directly with the Quran Expert assistant through a user-friendly Chat UI:
```bash
conda activate quran_llm
cd src
python app.py
```
> The web app will generally launch on `http://127.0.0.1:7860`.

### Option 2: FastAPI Server
To launch the RESTful backend endpoints (great for frontend/mobile app integrations):
```bash
conda activate quran_llm
cd src
uvicorn quran_api:app --reload --host 0.0.0.0 --port 8000
```
> Explore the interactive API documentation at `http://127.0.0.1:8000/docs`.

### Missing Vector Store Fallback (Ingestion from Scratch)
If you skip downloading the vector store from Google Drive, the first time you run the application it will dynamically authenticate with HuggingFace, download the `MohamedRashad/Quran-Tafseer` dataset, convert all verses and interpretations using OpenAI's `text-embedding-3-small` model, and build the persistent ChromaDB index locally. **This will take time and incur OpenAI API costs**. Once complete, the vector index will be saved to `./vector_store/chromadb_quran_tafsir` for instant startup on future executions.

## Architecture Guidelines
To modify the core Tafsir generation prompts, update the System Prompts contained within:
- `agent1_system_prompt` (Ayah parsing logic)
- `agent2_system_prompt` (Generation, restriction, hallucination-guarding logic) 
Within both `src/quran_rag_agent.py` and `src/quran_api.py`.
