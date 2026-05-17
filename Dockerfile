FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY docs_dir/ docs_dir/
COPY steps.txt .
COPY start.sh .

# Create vector store directory
RUN mkdir -p /app/doc_vector_store

# Fix Windows line endings and make executable
RUN dos2unix /app/start.sh && chmod +x /app/start.sh

# Expose ports
EXPOSE 8000 8501

# Environment variables
ENV DOCUMENTS_DIR=/app/docs_dir
ENV VECTOR_STORE_DIR=/app/doc_vector_store
ENV COLLECTION_NAME=document_collection
ENV MODEL_NAME=groq/llama-3.3-70b-versatile
ENV MODEL_TEMPERATURE=0.0
ENV CHAT_ENDPOINT_URL=http://localhost:8000/chat/answer

# Start services
CMD ["/app/start.sh"]