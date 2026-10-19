from typing import Any, List, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph

from app.utils import (
    LLM_TEMPERATURE,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    get_embeddings,
)

class RAGState(TypedDict, total=False):
    question: str              # User's question
    documents: List[Document]  # Retrieved documents
    context: str               # Formatted context from retrieved documents
    answer: str                # Generated answer to the question

def format_docs(docs: List[Any]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for question answering. "
            "Use only the retrieved context to answer the question. "
            "If the answer is not in the context, say you could not find it in the documents.",
        ),
        (
            "human",
            "Question: {question}\n\nRetrieved context:\n{context}",
        ),
    ]
)

class RAGPipeline:
    """RAG Pipeline with dependency injection for flexible configuration and testing"""

    def __init__(self):
        """Initialize RAG pipeline with all dependencies"""
        self.llm = ChatMistralAI(
            model=MISTRAL_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=512,
            api_key=MISTRAL_API_KEY,
        )
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=get_embeddings(),
            namespace=PINECONE_NAMESPACE,
        )
        self.rag_workflow = self.build_workflow()

    def build_workflow(self):
        """Build the RAG workflow graph"""
        graph_builder = StateGraph(RAGState)
        graph_builder.add_node("retrieve_documents", self.retrieve_documents)
        graph_builder.add_node("generate", self.generate_answer)

        # Add edges following the diagram
        graph_builder.add_edge(START, "retrieve_documents")
        graph_builder.add_edge("retrieve_documents", "generate")
        graph_builder.add_edge("generate", END)
        return graph_builder.compile()

    def retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant chunks from the Pinecone index."""
        query = state["question"]
        documents = self.vector_store.similarity_search(query, k=5)
        return {
            "documents": documents,
            "context": format_docs(documents),
        }
    
    def answer(self, question: str) -> str:
        """Answer a question using the RAG pipeline"""
        state: RAGState = {"question": question}
        result = self.rag_workflow.invoke(state)
        return result["answer"]

    def generate_answer(self, state: RAGState) -> RAGState:
        chain = generation_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "question": state["question"],
            "context": state.get("context", ""),
        })
        return {"answer": answer}

# Initialize singleton pipeline instance
rag_pipeline = RAGPipeline()
