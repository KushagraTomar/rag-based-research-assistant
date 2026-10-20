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
frontend/      React + Vite UI (PDF upload and Q&A)
```

## License

See [LICENSE](LICENSE).
