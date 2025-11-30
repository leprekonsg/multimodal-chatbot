"""
FastAPI Server
REST API for the multimodal RAG chatbot with embedded frontend.
"""
import os
import sys
import traceback
import json
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

# Load env vars before config
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from chatbot import chatbot
from ingestion import ingestion_pipeline
from retrieval import enhanced_retriever
from usage import tracker
from handoff import handoff_manager

# ===== Pydantic Models =====

class ChatRequest(BaseModel):
    """Chat request body."""
    message: str = Field(..., description="User's text message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    stream: bool = Field(False, description="Enable streaming response")
    user_id: Optional[str] = Field(None, description="User identifier")

class ChatResponseModel(BaseModel):
    """Chat response body."""
    message: str
    conversation_id: str
    sources: List[dict] = []
    confidence: float
    escalated: bool
    escalation_reason: Optional[str] = None
    source_images: List[dict] = []
    latency_ms: float

class IngestRequest(BaseModel):
    """Ingestion request for text."""
    text: str = Field(..., description="Text content to ingest")
    title: Optional[str] = Field(None, description="Document title")
    metadata: Optional[dict] = Field(None, description="Additional metadata")

class IngestResponse(BaseModel):
    """Ingestion response."""
    id: str
    type: str
    url: Optional[str] = None
    caption: Optional[str] = None

class SearchRequest(BaseModel):
    """Search request."""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of results")
    filter_type: Optional[str] = Field(None, description="Filter by type (image/text)")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    providers: dict

class UsageResponse(BaseModel):
    voyage_tokens: int
    voyage_pixels: int
    qwen_input_tokens: int
    qwen_output_tokens: int
    estimated_cost_usd: float

# ===== Lifespan =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    print("🚀 Starting Multimodal RAG Chatbot (V2 Integration)...")
    print(f"   Voyage API: {'configured' if config.voyage.api_key else 'NOT SET'}")
    print(f"   Qwen API: {'configured' if config.qwen.api_key else 'NOT SET'}")
    print(f"   Qdrant: {config.qdrant.url}")
    
    # Ensure directories exist
    os.makedirs(config.storage.local_path, exist_ok=True)
    
    # Initialize V2 Collection
    try:
        await ingestion_pipeline.vector_store.ensure_collection()
        print("✅ Multi-vector collection ready.")
    except Exception as e:
        print(f"❌ Qdrant Connection Failed: {e}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")
    await ingestion_pipeline.close()
    await enhanced_retriever.close()


# ===== App =====

app = FastAPI(
    title="Multimodal RAG Chatbot API",
    description="Vision-language chatbot with Voyage embeddings and hybrid retrieval",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
if os.path.exists(config.storage.local_path):
    app.mount(
        "/images",
        StaticFiles(directory=config.storage.local_path),
        name="images"
    )

# Serve static frontend
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(static_dir)),
        name="static"
    )


# ===== Frontend Routes =====

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse(content="Frontend not found", status_code=404)

@app.get("/ingest", response_class=HTMLResponse)
async def serve_ingest_page():
    ingest_path = Path(__file__).parent / "static" / "ingest.html"
    if ingest_path.exists():
        return FileResponse(ingest_path)
    return HTMLResponse(content="Ingestion page not found", status_code=404)


# ===== API Endpoints =====

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        providers={
            "handoff": handoff_manager.available_providers,
            "storage": config.storage.provider,
            "embedding": "voyage-multimodal-3"
        }
    )

@app.get("/usage", response_model=UsageResponse)
async def get_usage():
    stats = tracker.total
    cost = (
        (stats.voyage_tokens / 1_000_000) * 0.12 +
        (stats.qwen_input_tokens / 1_000) * 0.004 + 
        (stats.qwen_output_tokens / 1_000) * 0.012
    )
    return UsageResponse(
        voyage_tokens=stats.voyage_tokens,
        voyage_pixels=stats.voyage_pixels,
        qwen_input_tokens=stats.qwen_input_tokens,
        qwen_output_tokens=stats.qwen_output_tokens,
        estimated_cost_usd=round(cost, 4)
    )

@app.post("/chat", response_model=ChatResponseModel)
async def chat(request: ChatRequest):
    try:
        response = await chatbot.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            user_metadata={"user_id": request.user_id} if request.user_id else None
        )
        
        # Find conversation ID if not returned explicitly
        conv_id = request.conversation_id
        if not conv_id:
            for cid, ctx in chatbot.conversations.items():
                if ctx.messages and ctx.messages[-1]["content"] == response.message:
                    conv_id = cid
                    break
        
        return ChatResponseModel(
            message=response.message,
            conversation_id=conv_id or "",
            sources=[{
                "id": s.id,
                "type": s.type,
                "title": s.title,
                "url": s.url,
                "relevance_score": s.relevance_score
            } for s in response.sources],
            confidence=response.confidence,
            escalated=response.escalated,
            escalation_reason=response.escalation_reason,
            source_images=response.source_images,
            latency_ms=response.latency_ms
        )
    except Exception as e:
        print("❌ Error in /chat:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/multimodal", response_model=ChatResponseModel)
async def chat_multimodal(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None),
    conversation_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None)
):
    try:
        image_data = None
        if image:
            image_data = await image.read()
        
        response = await chatbot.chat(
            message=message,
            image_data=image_data,
            conversation_id=conversation_id,
            user_metadata={"user_id": user_id} if user_id else None
        )
        
        conv_id = conversation_id
        if not conv_id:
            for cid, ctx in chatbot.conversations.items():
                if ctx.messages and response.message in ctx.messages[-1]["content"]:
                    conv_id = cid
                    break
        
        return ChatResponseModel(
            message=response.message,
            conversation_id=conv_id or "",
            sources=[{
                "id": s.id,
                "type": s.type,
                "title": s.title,
                "url": s.url,
                "relevance_score": s.relevance_score
            } for s in response.sources],
            confidence=response.confidence,
            escalated=response.escalated,
            escalation_reason=response.escalation_reason,
            source_images=response.source_images,
            latency_ms=response.latency_ms
        )
    except Exception as e:
        print("❌ Error in /chat/multimodal:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for token in chatbot.chat(
            message=request.message,
            conversation_id=request.conversation_id,
            stream=True
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: IngestRequest):
    try:
        doc = await ingestion_pipeline.ingest_text(
            text=request.text,
            title=request.title,
            metadata=request.metadata
        )
        return IngestResponse(
            id=doc.id,
            type=doc.type.value,
            caption=doc.caption
        )
    except Exception as e:
        print("❌ Error in /ingest/text:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/image", response_model=IngestResponse)
async def ingest_image(
    image: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    import json
    try:
        image_data = await image.read()
        meta = json.loads(metadata) if metadata else None
        
        doc = await ingestion_pipeline.ingest_image(
            image_data=image_data,
            filename=image.filename,
            metadata=meta
        )
        return IngestResponse(
            id=doc.id,
            type=doc.type.value,
            url=doc.url,
            caption=doc.caption
        )
    except Exception as e:
        print("❌ Error in /ingest/image:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest file with streaming progress updates (NDJSON)."""
    import tempfile
    import uuid as uuid_mod
    import aiofiles
    import json
    
    async def process_generator():
        temp_path = None
        try:
            # 1. Upload Phase
            suffix = Path(file.filename).suffix
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"upload_{uuid_mod.uuid4()}{suffix}")
            
            async with aiofiles.open(temp_path, 'wb') as f:
                while content := await file.read(1024 * 1024):
                    await f.write(content)
            
            # 2. Processing Phase
            async for update in ingestion_pipeline.ingest_file_stream(temp_path, metadata={"filename": file.filename}):
                yield json.dumps(update) + "\n"
                
        except Exception as e:
            print("❌ Error in /ingest/file stream:")
            traceback.print_exc()
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    return StreamingResponse(process_generator(), media_type="application/x-ndjson")


@app.post("/search")
async def search(request: SearchRequest):
    try:
        result = await enhanced_retriever.retrieve(
            query_text=request.query,
            top_k=request.top_k,
            filter_type=request.filter_type
        )
        
        return {
            "documents": [
                {
                    "id": doc.id,
                    "type": doc.type,
                    "title": doc.source_display,
                    "url": doc.url,
                    "caption": doc.caption,
                    "text": doc.text[:500] if doc.text else None,
                    "score": doc.score,
                    "match_type": doc.match_type
                }
                for doc in result.documents
            ],
            "confidence": result.confidence,
            "query_intent": result.query_intent.value
        }
    except Exception as e:
        print("❌ Error in /search:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    chatbot.clear_conversation(conversation_id)
    return {"status": "cleared", "conversation_id": conversation_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)