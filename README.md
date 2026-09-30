# Enterprise_knowledge_assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions over your
organization's documents, with citations and calibrated confidence. It offers two
answering strategies: a fast **single-shot** RAG and a **ReAct agent** that reformulates
and searches multiple times — and will *offer to escalate* to ReAct when a single-shot
answer comes back weak.

## Quick start (run it with your own API key)

```bash
# 1. Clone and enter the repo
git clone git@github.com:kpareek/Enterprise_knowledge_assistant.git
cd Enterprise_knowledge_assistant

# 2. Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Add YOUR OWN key (never commit .env — it's git-ignored)
cp .env.example .env
#   then edit .env:
#   - Easiest / free:  RAG_MODE=free  + a free OpenRouter key (https://openrouter.ai/keys)
#   - Paid (OpenAI):   RAG_MODE=paid  + your OPENAI_API_KEY (https://platform.openai.com/api-keys)
#   See .env.example for both paths and a note on free-tier daily request caps.

# 4. Build the vector index from the sample documents
python -m ingestion.ingest

# 5. Run the assistant (chat UI on http://localhost:8501)
streamlit run app.py
```

**Optional — evaluation & report:**
```bash
python -m eval.evaluate                 # score the single-shot pipeline
python -m eval.evaluate --strategy react # score the ReAct agent
streamlit run eval/dashboard.py          # visualize results (port 8502)
python eval/build_report_html.py         # render SUMMARY/EVALUATION_REPORT to shareable HTML
```

> **Note on free tier:** OpenRouter's `:free` models share an account-wide daily request cap
> (~50/day without credit). A full 28-question eval needs ~200 requests, so either add a small
> OpenRouter/OpenAI balance or spread eval runs across days. The chat UI itself is fine on free.

________________________________________
Project File Guide — Enterprise Knowledge Assistant
A folder-by-folder, file-by-file reference for everyone on the team working with this codebase. For deeper design rationale, see docs/architecture.md; for setup instructions, see docs/deployment_guide.md.
________________________________________
Top-Level Files
File				Purpose
app.py				The Streamlit chatbot UI — what you actually run to talk to the assistant. Wires together the vector store, retriever, and generator; shows the free/paid mode badge, department/sensitivity/organization filters, and per-message citations + confidence scores.
config.py			Central settings for the entire project. Controls the free/paid mode switch (RAG_MODE), chunk size, retrieval parameters, confidence thresholds, and file paths. Nothing else in the codebase hardcodes these values — change behavior here, not scattered across files.
requirements.txt	Python package dependencies (pip install -r requirements.txt).
.env				Your actual local secrets and settings (API keys, RAG_MODE). Not committed to version control — every developer creates their own.
.env.example		Template showing what .env should contain, with placeholder values. Copy this to .env and fill in real values.
README.md			Project overview, quick-start instructions, and dataset description.
________________________________________
ingestion/ — Turns raw documents into searchable chunks
File				Purpose
__init__.py			Empty — marks this folder as a Python package so other files can import from it.
loaders.py			Format-specific document readers. One function per file type (load_pdf, load_docx, load_pptx, load_txt) that extracts text page-by-page or slide-by-slide, plus a header parser that reads the TITLE:/DEPARTMENT:/ORGANIZATION:/etc. metadata block every document starts with. 
					load_directory() walks a folder and dispatches each file to the right loader.
chunking.py			Splits extracted text into ~800-character chunks with 150-character overlap, breaking on paragraph/sentence boundaries rather than mid-word. Every chunk keeps its source filename, page number, and full metadata for citation purposes.
ingest.py			The script you actually run (python -m ingestion.ingest). Orchestrates the whole pipeline: walk the folder → hash each file → skip anything unchanged → load/chunk/embed anything new or changed → save the updated manifest. This is what makes ingestion "delta" — safe to re-run anytime without re-embedding your whole corpus.
manifest.py			Small utility that tracks "what have I already embedded" as a JSON file (data/ingestion_manifest_free.json or _paid.json), keyed by a SHA-256 hash of each file's contents. This is what ingest.py checks to decide what to skip.
________________________________________
rag/ — Retrieval and answer generation
File				Purpose
__init__.py			Empty, package marker.
llm_client.py		The single place that talks to OpenAI/Azure/OpenRouter. Routes every embedding and chat-completion call based on config.RAG_MODE: free mode uses a local embedding model + OpenRouter's free-tier chat model; paid mode uses OpenAI or Azure OpenAI. Switching modes anywhere else in the app is just a config change, not a code change, because everything else calls through this file.
vector_store.py		Wraps ChromaDB — storing chunks with their embeddings, querying by similarity, filtering by department/sensitivity/organization metadata, and deleting a file's old chunks when it changes (used by delta ingestion).
retriever.py		The retrieval pipeline: dense vector search → BM25 keyword-search fusion → cross-encoder reranking down to the final top few chunks that actually get sent to the LLM.
generator.py		Builds the grounded prompt from retrieved chunks, calls the LLM, and returns the answer with citations and a confidence score. Also contains the organization-disambiguation logic — if retrieved chunks span more than one organization and the question doesn't already name one, it asks for clarification before calling the LLM, rather than risking a blended or wrong-entity answer.
________________________________________
eval/ — Measuring answer and retrieval quality
File				Purpose
evaluate.py			Runs every question in eval_testset.json through the real pipeline and scores retrieval hit-rate, mean reciprocal rank, answer keyword coverage, confidence, and latency. Writes results to eval_results_free.json or _paid.json. Run with python -m eval.evaluate.
dashboard.py		A separate Streamlit app (streamlit run eval/dashboard.py) that visualizes the results evaluate.py produces — confidence distribution, per-question latency, and a drill-down into any individual question's retrieved sources and answer.
eval_testset.json	The actual test questions — not code, just data. Each entry has a question, which document it should be answered from (or null for questions the assistant should correctly refuse), and keywords the answer should contain.
________________________________________
docs/ — Project documentation
File				Purpose
architecture.md		Full system design writeup: architecture diagram, component-by-component rationale (why hybrid retrieval, why reranking, why delta ingestion), and the security/governance path from this prototype to a production deployment.
deployment_guide.md	Practical setup instructions: local install, switching between OpenAI/Azure/free mode, moving off ChromaDB to a managed vector store, Docker, and a production hardening checklist.
________________________________________
data/ — The actual documents and the vector index
Item				Purpose
sample_docs/		The main document corpus — HR/IT/Finance/Compliance/Operations/Training policies and reports, in PDF/DOCX/PPTX. This is what the assistant actually answers questions from.
external_enterpriserag_bench/	
					10 real documents from the independently published EnterpriseRAG-Bench benchmark (engineering runbooks/playbooks), kept as-is in .txt format since that's genuinely how they were sourced.
chroma_db/			The vector database itself — embeddings, chunk text, and metadata, written here by ChromaDB. Don't edit this by hand; it's entirely managed by ingest.py. Ephemeral in the sense that it can always be rebuilt from sample_docs/ + external_enterpriserag_bench/ via --reset.
ingestion_manifest_free.json/
ingestion_manifest_paid.json	
					The delta-ingestion tracking file described under manifest.py above — one per mode, since free and paid mode use separate vector collections.
________________________________________
venv/
Your local Python virtual environment (created via python3 -m venv venv). Contains installed packages — never edited directly, never committed to version control, and safe to delete and recreate at any time via pip install -r requirements.txt.
________________________________________
Quick Mental Model
data/ (documents)
   │
   ▼
ingestion/  ──(embed + chunk)──▶  data/chroma_db/ (vector index)
                                        │
                                        ▼
rag/  ──(retrieve + rerank + generate)──▶  app.py (chat UI)
   │
   └──▶  eval/ (measures how well the above actually works)

If you're new to the codebase: start by reading README.md, then docs/architecture.md for the "why," then this file for the "where."


	
	Running the project locally
	-------------------------------------------------------

	1. Check prerequisites
	Make sure Python 3.10+ is installed (python3 --version). 
	You'll also need pip. No Node.js or other runtime is required to run the app itself — that's only needed if you want to regenerate the document-generator scripts.

	Install Visual Studio

	2. Clone Repository
	Open Visual Studio
	Click on File --> Clone Repository
	Provide location as : https://github.com/vhanawa1/Enterprise_knowledge_assistant.git
	Click on Clone button
	
	2. Open a terminal in the project folder
	cd into the folder containing app.py, config.py, ingestion/, rag/, data/, etc. If you're using VS Code, open this folder and use the integrated terminal (Ctrl+`).
	
	Command:  cd .\Enterprise_knowledge_assistant\

	3. Create and activate a virtual environment
	Command: python3 -m venv venv, 
	
	then activate it using below commands: 
	Command: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

	(macOS/Linux) 
	Command: source venv/bin/activate 
		or 
	(Windows PowerShell)
	Command: venv\Scripts\Activate.ps1 
		You should see (venv) at the start of your prompt once it's active.

	4. Install dependencies
	Command: pip install -r requirements.txt 

	This installs openai, chromadb, streamlit, pypdf, python-docx, python-pptx, sentence-transformers, python-dotenv, and everything else the pipeline needs. 
	The first install can take a few minutes.
	
	5. Create your .env file
	Copy .env.example to .env in the project root. 
	Set RAG_MODE=free or RAG_MODE=paid. For free mode, add OPENROUTER_API_KEY=sk-or-... 
	(get one at openrouter.ai/keys, no card needed). For paid mode, add OPENAI_API_KEY=sk-... instead. 
	Double-check there's no stale RAG_MODE or API key already exported in your shell session — .env now overrides shell exports, but it's worth a fresh terminal to be safe.

	6. Confirm all your documents are in data/sample_docs/
	This should include your original corpus plus everything we've added since: Global_Employee_Handbook.pdf, Q3_FY2026_Financial_Report.pdf, the 19 Northwind Analytics gap-filling documents, and the 12 rebranded documents from the last two batches. Also confirm data/external_enterpriserag_bench/ has its 10 files if you're using that source too.
	
	7.	Run ingestion (first time: use --reset)
	Command: python -m ingestion.ingest --dir data/sample_docs --reset
		then 
	Command: python -m ingestion.ingest --dir data/external_enterpriserag_bench 
	(no --reset, so it adds to the same collection). 
		The --reset flag wipes any old/stale data and builds a clean manifest. First run in free mode will download the local embedding model (~90MB) — that's normal and only happens once. 
		On every run AFTER this first one, drop --reset — delta ingestion will only embed new or changed files.

	8. Launch the app
	Command: streamlit run app.py
	Your terminal will print a Local URL, typically http://localhost:8501 —  
	open that in your browser. The sidebar should show your current mode (🆓 FREE or 💳 PAID), department/sensitivity/organization filters, and how many chunks are indexed.



GitHub :

To push README.md to GitHub, run these commands in your repo directory (replace remote URL and branch as needed):
1.	If you haven't set a remote: git remote add origin https://github.com/vhanawa1/Enterprise_knowledge_assistant.git
2.	Stage, commit, and push: 
git add README.md 
git commit -m "Update README.md" 
git push -u origin main
Notes:
•	If your default branch is master, replace main with master.
•	If push fails, authenticate (SSH key or HTTPS credentials) or create the repo on GitHub first.

eg.
git add README.md; 
git commit -m "commit README.md file"