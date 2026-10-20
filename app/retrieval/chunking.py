from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.retrieval.utils import convert_section_headers

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_INDEX_DIMENSION,
    PINECONE_NAMESPACE,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from app.retrieval.embedding import get_embeddings

headers_to_split_on = [
    ("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3"),
]


def chunk_documents(documents: List[Document]) -> List[Document]:
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for doc in documents:
        # Convert section headers to markdown format
        markdown_text = convert_section_headers(doc.page_content)
        # Split the document into chunks based on markdown headers
        for chunk in markdown_splitter.split_text(markdown_text):
            chunk.metadata.update(doc.metadata)  # keep source file info
            if len(chunk.page_content) <= CHUNK_SIZE:
                chunks.append(chunk)
            else:
                # Further split large chunks into smaller ones
                chunks.extend(text_splitter.split_documents([chunk]))
    return chunks


def ensure_index() -> None:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            vector_type="dense",
            dimension=PINECONE_INDEX_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def ingest_pdf(pdf_path: str) -> int:
    """Load, chunk and upload a single PDF. Returns number of chunks indexed."""
    ensure_index()
    # Load pdfs
    documents = PyMuPDF4LLMLoader(
        file_path=str(pdf_path), 
        mode="single").load()
    # Chunk documents
    chunks = chunk_documents(documents)
    # Embed and upload to Pinecone
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        index_name=PINECONE_INDEX_NAME,
        namespace=PINECONE_NAMESPACE,
    )
    return len(chunks)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parents[2] / "data"
    for pdf in sorted(data_dir.glob("*.pdf")):
        print(f"{pdf.name}: {ingest_pdf(str(pdf))} chunks")
