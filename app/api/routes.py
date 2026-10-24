"""Routes for RAG API"""
import hashlib
import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import RAGRequest, RAGResponse
from app.db import registry
from app.retrieval.chunking import delete_vectors, ingest_document
from app.rag.rag import rag_pipeline

router = APIRouter(prefix="", tags=["rag"])

# Temporary upload directory
UPLOAD_DIR = "data/temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Research Assistant RAG API", "version": "1.0.0"}


@router.post("/ask", response_model=RAGResponse)
async def ask_question_endpoint(request: RAGRequest):
    """Ask a question to the RAG system"""
    try:
        answer = rag_pipeline.answer(request.question)
        return RAGResponse(answer=answer, source_type="local_pdf", cached=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF file"""
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save temporary file, named by content hash so concurrent uploads don't collide
        content = file.file.read()
        doc_hash = hashlib.sha256(content).hexdigest()
        temp_path = os.path.join(UPLOAD_DIR, f"{doc_hash}.pdf")
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        try:
            # Ingest PDF into vector store (plain def route, so FastAPI runs this in a worker thread)
            doc, ingested = ingest_document(temp_path, file.filename, doc_hash)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not ingested and doc["status"] == "processing":
            raise HTTPException(status_code=409, detail=f"PDF '{file.filename}' is already being indexed")

        message = (
            f"PDF '{file.filename}' uploaded and indexed successfully!"
            if ingested
            else f"PDF '{file.filename}' is already indexed."
        )
        return {
            "status": "success",
            "message": message,
            "filename": file.filename,
            "doc_id": doc_hash,
            "already_indexed": not ingested,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@router.get("/documents")
def get_documents():
    """List every PDF in the registry and its ingestion status"""
    return registry.list_documents()


@router.delete("/documents/{doc_id}")
def remove_document(doc_id: str):
    """Delete a PDF's chunks from Pinecone and its registry row"""
    doc = registry.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc["status"] == "processing":
        raise HTTPException(status_code=409, detail="Document is being indexed; try again later")
    # delete document chuncks from Pinecone
    delete_vectors(doc_id)
    # delete document from registry
    registry.delete_document(doc_id)
    return {"status": "success", "message": f"Deleted '{doc['filename']}'", "doc_id": doc_id}


# @router.get("/cache/stats")
# async def get_cache_stats():
#     """Get cache statistics and performance metrics"""
#     cache = get_cache()
#     stats = cache.get_stats()
#     return {
#         "status": "success",
#         "cache_stats": stats
#     }


# @router.post("/cache/clear")
# async def clear_cache():
#     """Manually clear the entire cache"""
#     cache = get_cache()
#     cache.clear("rag:answer:*")
#     logger.info("Cache manually cleared via API")
#     return {
#         "status": "success",
#         "message": "Cache cleared successfully"
#     }
