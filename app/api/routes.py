"""Routes for RAG API"""
import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import RAGRequest, RAGResponse
from app.retrieval.chunking import ingest_pdf
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
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF file"""
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save temporary file
        temp_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
                
        # Ingest PDF into vector store
        ingest_pdf(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return {
            "status": "success",
            "message": f"PDF '{file.filename}' uploaded and indexed successfully!",
            "filename": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


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
