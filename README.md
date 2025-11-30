# Multimodal RAG Chatbot

A production-ready chatbot with **Voyage Multimodal-3 embeddings** for superior document understanding, hybrid retrieval, and intelligent human escalation.

## ✨ Why Voyage Multimodal-3?

| Capability | CLIP-based Models | Voyage Multimodal-3 |
|------------|-------------------|---------------------|
| Tables/Charts | ~60% accuracy | ~85% accuracy (+40%) |
| Context Window | 77 tokens | 32,000 tokens |
| Cross-modal Bias | Present | Eliminated |
| Document Screenshots | Poor | Native support |
| Interleaved Text+Image | No | Yes |

Voyage's unified backbone processes all modalities together, eliminating the need for complex document parsing pipelines.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                       │
│                     (Text Query and/or Image Upload)                          │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
    ┌───────────────────────────┐   ┌───────────────────────────┐
    │     VOYAGE DENSE SEARCH   │   │     SPARSE SEARCH         │
    │   (Multimodal-3 API)      │   │   (BM25 on Captions)      │
    │                           │   │                           │
    │  1024-dim embeddings      │   │  Keyword matching         │
    │  input_type="query"       │   │  for fallback             │
    └─────────────┬─────────────┘   └─────────────┬─────────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │   RECIPROCAL RANK FUSION      │
                  │   score(d) = Σ 1/(k + rank)   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │       QWEN3-VL-PLUS           │
                  │   (RAG Response Generation)   │
                  └───────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                                       ▼
    ┌─────────────────────────┐         ┌─────────────────────────┐
    │   RESPOND TO USER       │         │   HUMAN HANDOFF         │
    │   + Source Citations    │         │   (Webhook/Slack)       │
    └─────────────────────────┘         └─────────────────────────┘
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Clone and setup
git clone <repo>
cd multimodal-rag-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys:
# - VOYAGE_API_KEY (required) - get from voyageai.com
# - DASHSCOPE_API_KEY (required) - get from Alibaba Cloud
```

### 3. Run

```bash
python server.py
```

Open http://localhost:8000 for the Knowledge Console UI.

## 📁 Project Structure

```
multimodal-rag-chatbot/
├── config.py          # Centralized configuration
├── embeddings.py      # Voyage Multimodal-3 client
├── llm_client.py      # Qwen3-VL for generation
├── ingestion.py       # Document/image ingestion
├── retrieval.py       # RRF hybrid search
├── escalation.py      # Human handoff logic
├── handoff.py         # Webhook/Slack integration
├── chatbot.py         # Main orchestration
├── server.py          # FastAPI endpoints
├── storage.py         # Image storage (S3/local)
├── static/
│   └── index.html     # Knowledge Console UI
├── requirements.txt
├── .env.example
└── README.md
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Knowledge Console UI |
| `/chat` | POST | Text-only chat |
| `/chat/multimodal` | POST | Chat with image upload |
| `/chat/stream` | POST | Streaming response |
| `/ingest/image` | POST | Add image to KB |
| `/ingest/text` | POST | Add text to KB |
| `/search` | POST | Search knowledge base |
| `/health` | GET | Health check |

## 🔑 Model Usage

| Task | Model | Purpose |
|------|-------|---------|
| Embeddings | Voyage Multimodal-3 | Dense retrieval (1024-dim) |
| Captioning | Qwen3-VL-Flash | Image → text for BM25 |
| Generation | Qwen3-VL-Plus | RAG response synthesis |
| Sentiment | Qwen-Turbo | Escalation detection |

## ⚡ Performance

| Metric | Target | How Achieved |
|--------|--------|--------------|
| TTFT | <2s | Voyage API, no local model loading |
| Retrieval accuracy | 90%+ | RRF fusion + Voyage precision |
| Tables/charts | 85%+ | Voyage's document understanding |

## 🎨 Frontend

The Knowledge Console features:
- **Dark theme** with warm amber accents
- **Split layout**: Chat + live source panel  
- **Distinctive typography**: Crimson Pro + IBM Plex Mono
- **Smooth animations** and micro-interactions
- **Mobile responsive** with collapsible sources

## 🚨 Escalation Flow

1. **Explicit request**: "talk to human" → Immediate handoff
2. **Low confidence**: <0.5 retrieval score → Handoff
3. **LLM uncertainty**: "I don't have info" → Handoff
4. **Negative sentiment**: Frustration detected → Handoff
5. **Repeated failures**: 2+ failed attempts → Handoff

## 📝 License

MIT
