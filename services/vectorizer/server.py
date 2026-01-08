from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import base64
import logging
import os
import struct
import uvicorn


# Configure logging.
log_level = os.getenv("LOG_LEVEL", "WARN").upper()
logging.basicConfig(level=getattr(logging, log_level))
logger = logging.getLogger(__name__)

app = FastAPI(title="GS Search Vectorizer Service")

# Load model at startup.
MODEL_NAME = os.getenv("MODEL_NAME")
logger.info(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbeddingData(BaseModel):
    embedding: str  # Base64 encoded bytes.
    dimension: int


class EmbedResponse(BaseModel):
    embeddings: List[EmbeddingData]


@app.post("/embed", response_model=EmbedResponse)
def embed_text(request: EmbedRequest):
    """
    Generate embedding vectors for the provided texts (batch endpoint).
    """
    if not request.texts or len(request.texts) == 0:
        raise HTTPException(status_code=400, detail="Texts array cannot be empty")
    
    # Validate that no text is empty.
    for i, text in enumerate(request.texts):
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail=f"Text at index {i} cannot be empty")
    
    try:
        # Generate embeddings in batch.
        embeddings = model.encode(request.texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Convert each embedding to bytes (FLOAT32 little-endian).
        result_embeddings = []
        for embedding in embeddings:
            embedding_bytes = b''.join(struct.pack('<f', float(x)) for x in embedding)
            embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
            result_embeddings.append(EmbeddingData(
                embedding=embedding_b64,
                dimension=len(embedding)
            ))
        
        return EmbedResponse(embeddings=result_embeddings)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dimension": model.get_sentence_embedding_dimension()
    }


@app.get("/")
async def root():
    """
    Root endpoint with service information.
    """
    return {
        "service": "GS Search Vectorizer",
        "model": MODEL_NAME,
        "dimension": model.get_sentence_embedding_dimension(),
        "endpoints": {
            "embed": "POST /embed - Generate embeddings",
            "health": "GET /health - Health check"
        }
    }


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        access_log=False
    )
