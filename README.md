# RAG Based Research Assistant

A Retrieval-Augmented Generation (RAG) assistant for answering questions about your PDFs. Upload a PDF, and the backend chunks it, embeds it with Mistral, and stores it in Pinecone. Questions are answered by a Mistral chat model using only the retrieved chunks.

- **Backend:** FastAPI, LangChain, LangGraph
- **LLM & embeddings:** Mistral AI (`ministral-3b-2512` for answers, `mistral-embed` for embeddings)
- **Vector store:** Pinecone (serverless, AWS `us-east-1`)
- **Frontend:** React + Vite, Axios

## How it works

1. **Ingestion** ([app/retrieval/chunking.py](app/retrieval/chunking.py)): PDFs are loaded as Markdown with PyMuPDF4LLM. Numbered section headings are turned into Markdown headers, then the text is split by header and any section longer than 1000 characters is split again (150-character overlap). The chunks are embedded and uploaded to the `pdf-structured-rag-mistral` Pinecone index, which is created on first use (1024 dimensions, cosine).
2. **Answering** ([app/rag/rag.py](app/rag/rag.py)): a two-node LangGraph workflow retrieves the top 5 chunks for the question, then asks the Mistral model to answer from that context only. If the answer isn't in the documents, the model says so.

Model names, chunk sizes, and index settings live in [app/config.py](app/config.py).

## Setup

Requirements: Python 3.10+, Node.js, a Mistral API key, and a Pinecone API key. Both services are hosted, so you also need an internet connection.

1. Install the backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your API keys:
   ```bash
   export MISTRAL_API_KEY="your-mistral-api-key"
   export PINECONE_API_KEY="your-pinecone-api-key"
   ```

3. Install the frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Run

Start the backend on port 8001 (the frontend expects this port):
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8001 --reload
```

In a second terminal, start the frontend (opens http://localhost:5173):
```bash
cd frontend
npm run dev
```

Upload a PDF from the UI, then ask questions about it.

### Bulk ingestion (optional)

To index many PDFs at once without the UI, put them in a `data/` folder at the project root and run:
```bash
python -m app.retrieval.chunking
```

If you change the embedding model, re-run ingestion (and update `PINECONE_INDEX_DIMENSION` if the vector size changes).

## Evaluation

The [evals/](evals/) folder scores the pipeline with [RAGAS](https://docs.ragas.io/), using `ministral-14b-2512` as the judge model. Run all commands from the project root with your API keys set, and index some PDFs first.

1. Generate a question set from the indexed chunks:
   ```bash
   python -m evals.generate_dataset --n 30
   ```
   This samples chunks from Pinecone, skipping short ones such as headers and references, and asks the judge model to write one question and reference answer per chunk. The output goes to `evals/dataset.jsonl`. Review it before running the eval, and generate it again whenever you re-run ingestion, because each row points to a chunk ID.

2. Run the eval:
   ```bash
   python -m evals.evals --limit 5   # quick check
   python -m evals.evals             # full dataset
   ```
   Each question goes through the real pipeline, then RAGAS scores the result from 0 to 1 (higher is better):

   | Metric | Component | Measures |
   |--------|-----------|----------|
   | `llm_context_precision_with_reference` | Retrieval | Whether the useful chunks are ranked near the top |
   | `context_recall` | Retrieval | Whether the retrieved chunks contain everything in the reference answer |
   | `faithfulness` | Generation | Whether every claim in the answer is supported by the retrieved chunks |
   | `answer_relevancy` | Generation | Whether the answer addresses the question |

   Each run writes to `evals/results/<timestamp>/`: `results.csv` has per-question scores, and `summary.json` has the averages plus the models, `k`, and dataset used.

The judge model has a low per-minute rate limit, so a full run is slow and may retry calls. Options: `--dataset`, `--limit`, and `--out-dir` for the eval, and `--n` and `--out` for dataset generation.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Welcome message and API version |
| `POST` | `/ask` | Ask a question about the indexed PDFs |
| `POST` | `/upload-pdf` | Upload a PDF (multipart field `file`) and index it |

Interactive docs are available at http://localhost:8001/docs while the server is running.

### `POST /ask`

Request:
```json
{
  "question": "What are the types of agent memory?"
}
```

Response:
```json
{
  "answer": "The types of agent memory include...",
  "documents": null,
  "source_type": "local_pdf"
}
```

### `POST /upload-pdf`

```bash
curl -F "file=@paper.pdf" http://localhost:8001/upload-pdf
```

Response:
```json
{
  "status": "success",
  "message": "PDF 'paper.pdf' uploaded and indexed successfully!",
  "filename": "paper.pdf"
}
```

## Project structure

```
app/
  api/         FastAPI app and routes
  models/      Pydantic request/response schemas
  rag/         LangGraph retrieve-then-generate pipeline
  retrieval/   PDF loading, chunking, embeddings, Pinecone ingestion
  config.py    Model, chunking, and index settings
evals/         RAGAS evaluation: dataset generation and scoring
frontend/      React + Vite UI (PDF upload and Q&A)
```

## License

See [LICENSE](LICENSE).
