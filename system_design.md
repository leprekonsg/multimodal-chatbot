
# System Design Document: Multimodal RAG Chatbot (v2)

## 1. Executive Summary
This document details the architecture of a production-grade Multimodal Retrieval-Augmented Generation (RAG) system. The system is designed to ingest, index, and retrieve complex visual documents (technical manuals, schematics, charts) and provide grounded answers using Vision-Language Models (VLMs) with stateful multi-turn conversations.

**Key Differentiators:**
*   **Multi-Vector Architecture:** Decouples visual and textual semantic representations.
*   **Hybrid Retrieval:** Combines Dense, Sparse (BM25), and Perceptual Hash search strategies via Reciprocal Rank Fusion (RRF).
*   **Visual Grounding:** Native support for bounding box detection and coordinate mapping for UI highlighting.
*   **Stateful Conversations:** Turn-based context with 3-turn image retention, pronoun-aware query rewriting, and automatic token budgeting (28k limit).
*   **Active Escalation:** Deterministic and sentiment-based triggers for human agent handoff.

---

## 2. System Architecture

The system follows a microservices-lite architecture powered by **FastAPI** (AsyncIO), utilizing **Qdrant** for vector storage, **Voyage AI** for multimodal embeddings, and **Alibaba Qwen** for reasoning. Conversations are stateful with turn-based context management and token budgeting.

### 2.1 High-Level Data Flow

```mermaid
graph TD
    Client[Web Client / API] -->|HTTP/SSE| Server[FastAPI Server]

    subgraph Orchestration
        Server --> Chatbot[Chatbot Orchestrator]
        Chatbot --> Context[ConversationContext]
        Chatbot --> Escalation[Escalation Engine]
    end

    subgraph "Knowledge Layer"
        Chatbot --> Retriever[Hybrid Retriever]
        Retriever -->|Query| Embed[Voyage AI Embedder]
        Retriever -->|Search| Qdrant[(Qdrant Vector DB)]
        Retriever -->|Refine| Rerank[Voyage Reranker]
    end

    subgraph "Reasoning Layer"
        Chatbot --> LLM[Qwen3-VL Client]
        Context -->|History+Images| LLM
        LLM -->|Generate| Response
    end

    subgraph "Storage Layer"
        Server --> ObjStore[Object Storage S3/Local]
    end

    Escalation -->|Trigger| Handoff[Slack/Webhook]
```

---

## 3. Ingestion Subsystem

The ingestion pipeline transforms raw files (PDF, Images) into a multi-vector representation optimized for asymmetric retrieval (Image-to-Text, Text-to-Image).

### 3.1 Multi-Vector Strategy
Unlike standard RAG which stores one vector per document, this system stores **three dense vectors** and **one sparse vector** per asset to maximize retrieval surface area.

| Vector Name | Model | Dimension | Purpose |
| :--- | :--- | :--- | :--- |
| `image_dense` | Voyage Multimodal-3 | 1024 | Pure visual similarity search. |
| `text_dense` | Voyage Multimodal-3 | 1024 | Semantic search on captions/OCR. |
| `combined_dense` | Voyage Multimodal-3 | 1024 | Fused multimodal representation. |
| `sparse` | Custom BM25 | Dynamic | Keyword/Part-number exact matching. |

### 3.2 Contextual Enrichment & Grounding
Before embedding, images undergo **Structured Captioning** via `Qwen3-VL-Flash`.
1.  **Component Detection:** Extracts bounding boxes (`bbox_2d`) for key elements (valves, buttons).
2.  **Contextual Prefixing:** Prepends metadata (Source, Page, Type) to the text chunk to resolve ambiguity (Anthropic Contextual Retrieval technique).
3.  **Perceptual Hashing:** Computes pHash/dHash for O(1) exact duplicate detection.

### 3.3 Ingestion Flow

```mermaid
sequenceDiagram
    participant API as API
    participant Pipe as IngestionPipeline
    participant VLM as Qwen VLM
    participant Embed as Voyage Embedder
    participant DB as Qdrant

    API->>Pipe: Upload Image/PDF
    Pipe->>Pipe: Preprocess (Resize/Normalize)
    Pipe->>VLM: Request Structured Caption
    VLM-->>Pipe: JSON {description, components[], topics}
    
    par Parallel Embedding
        Pipe->>Embed: Encode Image Only
        Pipe->>Embed: Encode Text (Caption + Prefix)
        Pipe->>Embed: Encode Combined
    end
    
    Embed-->>Pipe: Return 3x Vectors
    
    Pipe->>Pipe: Compute Sparse Vector (BM25)
    Pipe->>Pipe: Compute Perceptual Hash
    
    Pipe->>DB: Upsert Point (Vectors + Payload + BBoxes)
```

---

## 4. Retrieval Subsystem

The retrieval engine employs an **Adaptive Query** strategy. It classifies the user intent to dynamically weight different search strategies.

### 4.1 Query Classification
Incoming queries are classified into intents:
*   `VISUAL_SEARCH`: "What is this?" (Image provided)
*   `TEXTUAL_SEARCH`: "How do I reset the pump?"
*   `EXACT_MATCH`: "Find this specific diagram."

### 4.2 Hybrid Search & Fusion
The system executes parallel searches based on intent. Results are aggregated using **Weighted Reciprocal Rank Fusion (RRF)**.

*   **Visual Query:** Weights `image_dense` (2.0) and `exact_hash` (3.0) higher.
*   **Text Query:** Weights `text_dense` (2.0) and `sparse` (1.5) higher.

### 4.3 Reranking (Precision Layer)
Top-K candidates (default 50) from RRF are passed to a **Cross-Encoder (Voyage Rerank 2)**.
*   **Input:** Query + Document Text (Caption + OCR).
*   **Output:** Re-ordered list based on deep semantic relevance.
*   **Impact:** Filters out "visually similar but semantically irrelevant" results.

### 4.4 Retrieval Flow

```mermaid
graph TD
    Query[User Query] --> Classifier{Intent Classifier}
    
    Classifier -->|Text| TextPath[Text Search]
    Classifier -->|Image| VisualPath[Visual Search]
    
    subgraph "Parallel Execution"
        TextPath --> Sparse[Sparse Search BM25]
        TextPath --> DenseText[Dense Text Search]
        
        VisualPath --> DenseVis[Dense Visual Search]
        VisualPath --> Hash[Perceptual Hash Search]
    end
    
    Sparse --> Fusion[Weighted RRF Fusion]
    DenseText --> Fusion
    DenseVis --> Fusion
    Hash --> Fusion
    
    Fusion --> Candidates[Top-50 Candidates]
    Candidates --> Reranker[Voyage Cross-Encoder]
    Reranker --> Final[Top-K Context]
```

---

## 5. Multi-Turn Conversation Management

The system maintains stateful conversations with automatic context management, image retention, and token budgeting aligned with Qwen3-VL-Plus 32k context window.

### 5.1 ConversationContext Architecture
**Storage:** In-memory Dict (production: Redis recommended)
**Token Limit:** 28,000 tokens (safety buffer below 32k)
**Warning Threshold:** 24,000 tokens
**Tracking:** Turn numbers, image uploads, API token usage

### 5.2 Image Retention Policy
**Rule:** Images retained for 3 turns after upload
**Budget:** Max 2 user images + 3 KB images = ~9k tokens (32% of budget)
**Rationale:** Balance accuracy for immediate follow-ups vs. conversation length

**Example Flow:**
- Turn 1: User uploads `pump.jpg` → Image passed to VLM ✓
- Turn 2-4: "Where is valve?" → Image passed to VLM ✓
- Turn 5+: "Maintenance schedule?" → Image excluded, use caption ✗

### 5.3 Query Rewriting
**Trigger:** Pronoun detection (`it|that|them|these|those`)
**Method:** Lightweight LLM rewrite using last 6 messages as context
**Fallback:** Original query if rewrite fails
**Cost:** ~80 tokens per rewrite (30% of follow-ups)

### 5.4 Token Management
**API-Based Counting:** Uses actual `response.usage` from Qwen API, not character estimates
**Preemptive Pruning:** Triggers when:
- Message count > 40 (20 turn pairs)
- Estimated tokens > 85% of max (24k)

**Reactive Pruning:** Auto-removes oldest message pairs when exceeding 28k
**User Notice:** Subtle inline warning: "Earlier messages hidden to manage context"

### 5.5 Conversation Lifecycle
**Creation:** Auto-generated UUID on first message
**Persistence:** Until explicit "New Chat" or server restart (in-memory)
**Cleanup:** No auto-expiry (manual only)

---

## 6. Generation & Visual Grounding

The generation layer uses `Qwen3-VL-Plus` to synthesize answers. It is context-aware and capable of **Visual Grounding** (locating elements in images).

### 6.1 Visual Grounding API
The system exposes a dedicated endpoint `/visual-grounding` that resolves natural language queries to image coordinates.
1.  **Check Index:** Looks for pre-computed bounding boxes in Qdrant metadata.
2.  **Real-time Inference:** If not indexed, invokes Qwen3-VL to predict `bbox_2d` on the fly.
3.  **Normalization:** Converts model coordinates (0-1000) to frontend percentages.

### 6.2 Streaming Response
Responses are streamed via Server-Sent Events (SSE) to minimize Time-To-First-Token (TTFT). The stream includes:
1.  Text tokens.
2.  Source citations (Markdown).
3.  Visual grounding metadata (for UI overlays).

---

## 7. Escalation Engine

The system includes a deterministic state machine to handle failures and handoffs. Integrated with ConversationContext for multi-turn failure tracking.

### 7.1 Triggers
| Trigger Type | Condition | Action |
| :--- | :--- | :--- |
| **Explicit** | User says "talk to human", "agent" | Immediate Handoff |
| **Confidence** | Retrieval Confidence < 0.50 | Soft Escalation Offer |
| **Sentiment** | Sentiment Score < -0.6 | Priority Handoff |
| **Failure** | 2+ Consecutive Low Confidence responses | Handoff |

### 7.2 Confidence Calibration
Confidence is not raw vector similarity. It is a calculated metric:
$$ C = S_{top} + \text{GapBonus}(S_{top} - S_{2nd}) + \text{IntentBonus} + \text{ExactMatchBonus} $$
*   $S_{top}$: Cosine similarity of top result.
*   GapBonus: Rewards distinct answers.

---

## 8. Data Schema (Qdrant)

The vector database schema is designed for flexibility and speed.

**Collection Config:**
```json
{
  "vectors": {
    "image_dense": { "size": 1024, "distance": "Cosine" },
    "text_dense": { "size": 1024, "distance": "Cosine" },
    "combined_dense": { "size": 1024, "distance": "Cosine" }
  },
  "sparse_vectors": {
    "sparse": { "index": { "on_disk": false } }
  }
}
```

**Payload Schema:**
```json
{
  "id": "uuid",
  "type": "image|text|pdf",
  "url": "s3://bucket/path/img.jpg",
  "caption": "Full generated caption...",
  "extracted_text": "OCR text content...",
  "fingerprint": {
    "phash": "binary_string",
    "avg_color": [r, g, b]
  },
  "components": [
    {
      "label": "Pressure Valve",
      "bbox_2d": [100, 200, 150, 250],
      "type": "valve"
    }
  ],
  "metadata": {
    "filename": "manual.pdf",
    "page_number": 12
  }
}
```

---

## 9. Infrastructure & Scalability

*   **Conversation State:** In-memory Dict for development; Redis recommended for production multi-instance deployment.
*   **Async I/O:** All external calls (Voyage, Qwen, Qdrant, S3) are non-blocking `async/await`.
*   **Rate Limiting:** Client-side semaphores enforce strict rate limits (e.g., 3 RPM for embedding) to manage API costs/quotas.
*   **Storage Abstraction:** `storage.py` provides a unified interface for Local, S3, and OSS, allowing seamless cloud migration.
*   **Token Budget:** Strict 28k limit with preemptive pruning prevents over-budget API calls and cost overruns.