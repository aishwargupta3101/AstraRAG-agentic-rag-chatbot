FROM python:3.11-slim

WORKDIR /app

#Install system dependenices
RUN apt-get update &&apt-get install -y\
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#copy requirements and install \
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

COPY src/ src/
COPY docs_dir/ docs_dir/
COPY steps.txt ./
COPY start.sh ./start.sh

# Make start.sh exectable
RUN chmod +x start.sh

#expose backend and fronted ports
EXPOSE 8000 8501

#set environment variables (can be overridden at runtime)
ENV DOCUMENTS_DIR="/app/docs_dir"
ENV VECTOR_STORE_DIR="/app/doc_vector_store"
ENV COLLECTION_NAME="document_collection"
ENV MODEL_NAME="llama-3.3-70b-versatile"
ENV MODEL_TEMPERATURE=0.0
ENV CHAT_ENDPOINT_URL="http://localhost:8000/chat/answer"

# RUn all services using start.sh
CMD ["/app/start.sh"]