# Quick Reference: Voyage Multimodal-3 Integration

## 🔄 Changes from Original

| Component | Before | After |
|-----------|--------|-------|
| **Embeddings** | Jina CLIP v2 (local) | Voyage Multimodal-3 (API) |
| **Dimensions** | 768 | 1024 |
| **Dependencies** | sentence-transformers, torch | httpx (lightweight) |
| **Image handling** | Local preprocessing | API handles resize |

## 🔑 API Keys Required

```bash
# Voyage AI - for embeddings
VOYAGE_API_KEY=your_key

# Alibaba Cloud - for Qwen VL
DASHSCOPE_API_KEY=your_key
```

## 📊 Voyage API Format

### Text Embedding
```python
await voyage_embedder.encode_text(
    "Your query here",
    input_type="query"  # or "document"
)
```

### Image Embedding
```python
await voyage_embedder.encode_image(
    image_bytes,
    input_type="document"
)
```

### Multimodal (Text + Image)
```python
await voyage_embedder.encode_multimodal(
    text="Caption or description",
    image=image_bytes,
    input_type="document"
)
```

## 🎯 Voyage Input Types

| Context | input_type | Effect |
|---------|-----------|--------|
| Indexing docs/images | `"document"` | Prepends document prompt |
| User queries | `"query"` | Prepends query prompt |
| Direct embedding | `None` | No prompt prepended |

## ⚡ Quick Commands

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Run server
python server.py

# Ingest image
curl -X POST http://localhost:8000/ingest/image \
  -F "image=@chart.png"

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does the chart show?"}'
```

## 🎨 Frontend Design

**Typography:**
- Headers: Crimson Pro (elegant serif)
- Body/Code: IBM Plex Mono (distinctive mono)

**Colors:**
- Background: `#0A0A0C` (deep charcoal)
- Accent: `#D4A574` (warm amber)
- Secondary: `#6B8B6B` (sage green)

**Key Features:**
- Split panel layout
- Live source cards with images
- Typewriter-style responses
- Mobile-responsive

## 🔧 Configuration

```python
# config.py key settings

# Voyage (1024-dim embeddings)
voyage.dimension = 1024
voyage.model_name = "voyage-multimodal-3"
voyage.max_image_pixels = 16_000_000

# Qdrant (updated for 1024-dim)
qdrant.collection_name = "knowledge_base"

# Escalation thresholds
escalation.low_confidence_threshold = 0.5
escalation.warn_confidence_threshold = 0.6
```

## 📁 File Changes Summary

| File | Status | Notes |
|------|--------|-------|
| `config.py` | Updated | Added VoyageConfig, updated dimension |
| `embeddings.py` | Replaced | Voyage API instead of CLIP |
| `ingestion.py` | Updated | Async Voyage calls |
| `retrieval.py` | Updated | Async Voyage calls |
| `server.py` | Updated | Frontend serving, cleanup |
| `requirements.txt` | Updated | Removed torch/sentence-transformers |
| `static/index.html` | New | Knowledge Console UI |
| Other files | Unchanged | chatbot, escalation, handoff, storage, llm_client |

## 🚀 Deployment Notes

1. **No GPU required** - Voyage is API-based
2. **Lighter dependencies** - No torch, no transformers
3. **Faster cold start** - No model loading
4. **Cost: ~$0.12/1M tokens** - Free tier available for prototyping
