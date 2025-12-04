# Multimodal RAG Chatbot

Production-grade RAG system using **Voyage Multimodal-3** embeddings and **Qwen3-VL** vision-language models for technical documents, diagrams, and schematics. Zero GPU requirements.

## Key Features

*   **Multi-Vector Retrieval:** 3 dense vectors (image/text/combined) + sparse (BM25) + perceptual hash per document
*   **Query-Side Enrichment:** Captions user images with Qwen3-VL-235B before embedding for semantic matching
*   **Hybrid Search:** Weighted RRF fusion with Voyage Rerank-2 cross-encoder (graceful fallback)
*   **Visual Grounding:** Bounding box detection (0-1000 coords) with relevance scoring and toggle UI
*   **Multi-Turn Context:** 3-turn image retention, pronoun-aware query rewriting, 28k token budget
*   **Carousel Navigation:** Multi-image modal with thumbnails, keyboard shortcuts, and 8-bit FF aesthetic
*   **Zero-GPU:** API-based (Voyage AI + Alibaba Cloud), runs on standard CPUs
*   **Streaming SSE:** Async pipeline with early metadata for responsive UX

## Architecture

```
Ingestion: File → Qwen Caption+BBox → Voyage Multi-Vector → Qdrant
Retrieval: Query → [Enrichment?] → Hybrid Search → RRF → Reranker → Top-K
Generation: User Images + KB Images + Context → Qwen3-VL-Plus → Streamed Response
```

**Stack:** FastAPI (AsyncIO), Qdrant (vector DB), Voyage AI (embeddings), Qwen3-VL (VLM)

## Quick Start

### Prerequisites
*   Python 3.10+
*   Docker (for Qdrant)
*   API Keys: [Voyage AI](https://www.voyageai.com/), [Alibaba Cloud DashScope](https://dashscope.aliyun.com/)

### Installation

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
Create `.env`:

```ini
VOYAGE_API_KEY=your-voyage-key
DASHSCOPE_API_KEY=sk-your-qwen-key
QDRANT_URL=http://localhost:6333  # Optional
STORAGE_PROVIDER=local             # Optional: local|s3
```

### Run

```bash
docker run -d -p 6333:6333 qdrant/qdrant  # Start Qdrant
python server.py                          # Start server
```

Access at `http://localhost:8000`

## Technical Details

### Multi-Vector Storage
Each document stores 4 vectors in Qdrant:
- **image_dense** (1024d): Pure visual features
- **text_dense** (1024d): Caption + OCR + component names
- **combined_dense** (1024d): Fused multimodal representation
- **sparse** (BM25): Keyword/part number matching

### Query Enrichment
Generic image queries trigger caption generation:
- Detects: short query (< 20 words) + vague terms ("this", "that")
- Uses: Qwen3-VL-235B-Instruct (235B params) for accuracy
- Appends: `{query}\n\n[Image shows: {caption}]`
- Impact: +40-60% confidence for generic queries

### Models
| Component | Model | Provider |
|-----------|-------|----------|
| Embedding | voyage-multimodal-3 (1024d) | Voyage AI |
| Generation | qwen3-vl-plus (32k context) | Alibaba Cloud |
| Query Caption | qwen3-vl-235b-instruct | Alibaba Cloud |
| Ingestion Caption | qwen3-vl-flash | Alibaba Cloud |
| Reranking | voyage-rerank-2 | Voyage AI |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/multimodal` | POST | Chat with optional image (multipart form) |
| `/ingest/file` | POST | Ingest PDF/images (streaming NDJSON) |
| `/search` | POST | Debug retrieval with `{query, top_k}` |
| `/visual-grounding` | POST | Locate elements via bbox coordinates |
| `/ux/config` | GET | Get UI configuration |

## UI Features

### 8-Bit Final Fantasy Theme
- **Fonts:** Press Start 2P (headers), VT323 (body)
- **Colors:** Deep blue (#0C1445) + gold (#FFD54F) accents
- **Animations:** Stepped timing (no easing) for authentic 8-bit feel
- **Scanlines:** CRT effect overlay

### Multi-Image Carousel
- **Navigation:** Arrow buttons, thumbnails, keyboard (← → or 1-9)
- **Bounding Boxes:** Toggle visibility, positioned outside boxes
- **Relevance:** Opacity scales with component relevance scores

### Conversation Features
- **Image Retention:** 3-turn window for context
- **Query Rewriting:** Auto-resolves pronouns using conversation history
- **Token Budget:** 28k limit with auto-pruning

## Project Structure

```
├── server.py              # FastAPI server + endpoints
├── chatbot.py             # Orchestration layer
├── retrieval.py           # Hybrid search + RRF fusion
├── ingestion.py           # Multi-vector pipeline
├── llm_client.py          # Qwen3-VL integration
├── embeddings.py          # Voyage API client
├── reranking.py           # Voyage Rerank-2
├── escalation.py          # Handoff logic
├── config.py              # Centralized configuration
├── static/
│   ├── index.html         # Single-page UI
│   ├── css/chat.css       # FF-themed styling
│   └── js/chat.js         # Carousel + streaming logic
└── system_design.md       # Architecture documentation
```

## Documentation

- **System Design:** See [system_design.md](system_design.md) for detailed architecture
- **Configuration:** Model tiers, thresholds, and settings in `config.py`
- **API Usage:** Track token consumption via `/usage` endpoint