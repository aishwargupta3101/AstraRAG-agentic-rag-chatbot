import logging
import chromadb

from crewai.tools import tool
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.groq import Groq
from src.agents_src.config.agent_settings import AgentSettings

logger = logging.getLogger(__name__)# this will tell you where the error came from

logger.info("Loading HuggingFace embedding model...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en"
)

@tool
def rag_query_tool(query: str) -> dict:
    """
    Query the vector database and return answer with sources.
    """

    try:
        settings = AgentSettings()

        # LLM
        Settings.llm = Groq(
            model=settings.MODEL_NAME,
            temperature=settings.MODEL_TEMPERATURE,
            api_key=settings.GROQ_API_KEY
        )
        Settings.embed_model = embed_model
        db = chromadb.PersistentClient(
            path=settings.VECTOR_STORE_DIR
        )
        chroma_collection = db.get_or_create_collection(
            settings.COLLECTION_NAME
        )
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        # Load index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        # Query engine
        query_engine = index.as_query_engine(
            similarity_top_k=5
        )
        response = query_engine.query(query)
        source_files = set()
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                print(node.text)
                metadata = node.node.metadata
                if metadata and "file_name" in metadata:
                    source_files.add(
                        metadata["file_name"]
                    )
        return {
            "answer": str(response),
            "sources_file": list(source_files)
        }
    except Exception as e:
        logger.error(f"RAG tool error: {e}")
        return {
            "answer": "Error while retrieving answer.",
            "sources_file": [],
            "error": str(e)
        }