import logging
from crewai.tools import tool

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

import chromadb

from src.agents_src.config.agent_settings import AgentSettings

# Logger
logger = logging.getLogger(__name__)

# Load embedding model (only once)
logger.info("Loading HuggingFace embedding model...")
embed_model = HuggingFaceEmbedding()

@tool
def rag_query_tool(query: str) -> dict:
    """
    Answer a query by retrieving relevant documents and generating a response.
    Returns both answer and source file names.
    """

    try:
        settings = AgentSettings()
        # DEBUG (you can remove later)
        print("DEBUG SETTINGS:", settings.model_dump())
        vector_store_path = settings.VECTOR_STORE_DIR
        collection_name = settings.COLLECTION_NAME
        # ✅ Configure LLM
        Settings.llm = Groq(
            model=settings.MODEL_NAME,
            temperature=settings.MODEL_TEMPERATURE,
            api_key=settings.GROQ_API_KEY
        )
        Settings.embed_model = embed_model
        db = chromadb.PersistentClient(path=vector_store_path)
        chroma_collection = db.get_or_create_collection(collection_name)
        # ✅ Connect vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        query_engine = index.as_query_engine(similarity_top_k=3)
        # ✅ Run query
        response = query_engine.query(query)
        source_files = set()
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                metadata = node.node.metadata
                if metadata and "file_name" in metadata:
                    source_files.add(metadata["file_name"])
        return {
            "answer": str(response),
            "sources_file": list(source_files)
        }
    except Exception as e:
        logger.error(f"RAG tool error: {e}")

        return {
            "answer": "Error occurred while retrieving answer.",
            "sources_file": [],
            "error": str(e)
        }