import logging
import os

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import SimpleDirectoryReader

from src.rag_doc_Ingestion.config.doc_Ingestion_settings import DocIngestionSettings

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# ---------------- SETTINGS ---------------- #
settings = DocIngestionSettings()
logger.info("Loading HuggingFace embedding model...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
# ---------------- MAIN FUNCTION ---------------- #
def build_vector_store_from_documents():
    try:
        logger.info("Starting vector store ingestion process...")

        docs_dir_path = settings.DOCUMENTS_DIR
        vector_store_path = settings.VECTOR_STORE_DIR
        collection_name = settings.COLLECTION_NAME

        # ✅ Ensure directory exists
        os.makedirs(vector_store_path, exist_ok=True)
        logger.info(f"Documents dir: {docs_dir_path}")
        logger.info(f"Vector store dir: {vector_store_path}")
        # ---------------- LOAD DOCUMENTS ---------------- #
        logger.info("Loading documents...")
        documents = SimpleDirectoryReader(docs_dir_path).load_data()

        # ---------------- CHUNKING ---------------- #
        parser = SimpleNodeParser.from_defaults(
            chunk_size=1024,
            chunk_overlap=50
        )
        logger.info("Parsing documents into nodes...")
        nodes = parser.get_nodes_from_documents(documents)
        logger.info(f"Parsed {len(nodes)} nodes.")
        # ---------------- CHROMADB (PERSISTENT) ---------------- #
        logger.info(f"Initializing persistent ChromaDB at: {vector_store_path}")
        client = chromadb.PersistentClient(path=vector_store_path)
        chroma_collection = client.get_or_create_collection(
            name=collection_name
        )
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        # ---------------- BUILD INDEX ---------------- #
        logger.info("Building vector store index...")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model
        )
        storage_context.persist()
        logger.info("✅ Vector store built and persisted successfully!")
        logger.info(f"📁 Stored at: {vector_store_path}")
    except Exception as e:
        logger.error(f"❌ Error during ingestion: {str(e)}")
        raise

if __name__ == "__main__":
    build_vector_store_from_documents()