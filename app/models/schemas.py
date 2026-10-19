"""Pydantic models for request/response validation"""
from typing import List, Optional
from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """Request model for RAG query"""
    question: str = Field(..., description="The user's question")


class RAGResponse(BaseModel):
    """Response model for RAG answer"""
    answer: str = Field(..., description="The generated answer")
    documents: Optional[List[dict]] = Field(None, description="Retrieved documents")
    source_type: Optional[str] = Field(None, description="Type of source: local_pdf or web_search")
    # cached: bool = Field(False, description="Whether the response was retrieved from cache")
