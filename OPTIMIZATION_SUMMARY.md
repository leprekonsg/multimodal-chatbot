# RAG System Optimization Summary

## Executive Summary

Based on the research evidence, three high-impact optimizations were implemented:

| Optimization | Research Evidence | Expected Impact |
|-------------|-------------------|-----------------|
| **Voyage Rerank 2** | 20-48% accuracy improvement (Databricks, Coalfire) | **Highest** |
| **Contextual Retrieval** | 67% error reduction (Anthropic) | **High** |
| **Visual Grounding** | Unique Qwen3-VL capability | **High UX** |

---

## Implementation Details

### 1. Voyage Rerank Integration (NEW FILE: `reranking.py`)

**What it does:**
- Cross-encoder reranking using Voyage Rerank 2 API
- Retrieves 50 candidates → Reranks → Returns top 5 to LLM
- Graceful degradation if reranking fails

**Research justification:**
- Databricks: Up to 48% improvement when combining hybrid search with reranking
- Coalfire production case: Hit rate increased from 58% to 87%
- ~300-600ms latency overhead, justified by quality gains

**Configuration (`config.py`):**
```python
@dataclass
class RerankConfig:
    enabled: bool = True
    model_name: str = "rerank-2"
    top_k: int = 5  # Final results to LLM
    candidates_to_rerank: int = 50  # From hybrid search
```

**Integration point (`retrieval.py`):**
```python
# After RRF fusion, before returning
if query_text and len(fused) > 1:
    fused = await reranker.rerank(
        query=query_text,
        documents=fused,
        top_k=top_k
    )
```

---

### 2. Contextual Retrieval (`ingestion.py`, `llm_client.py`)

**What it does:**
- Prepends chunk-specific context before embedding
- Example: `[Source: Pump Manual, Page 12] [Type: Schematic] [Topics: hydraulic, valve]`
- Dramatically improves retrieval for entity disambiguation

**Research justification:**
- Anthropic research: 49% reduction with contextual embeddings
- 67% reduction when combined with reranking
- Particularly effective for technical manuals with cross-references

**Implementation:**
```python
# In llm_client.py - generates context prefix
def _generate_context_prefix(self, structured, filename, page_number):
    parts = []
    if filename:
        parts.append(f"[Source: {source}]")
    if doc_type:
        parts.append(f"[Type: {type_label}]")
    if topics:
        parts.append(f"[Topics: {', '.join(topics[:3])}]")
    return " ".join(parts) + " "
```

---

### 3. Visual Grounding (`llm_client.py`, `server.py`)

**What it does:**
- Extracts component bounding boxes during ingestion
- Enables query-time localization: "Where is the reset button?"
- Returns normalized coordinates [0-1000] for frontend rendering

**Why it matters for technicians:**
- Can locate specific components on complex diagrams
- Reduces time-to-find for maintenance tasks
- Unique capability of Qwen3-VL (97% DocVQA accuracy)

**New API endpoint:**
```
POST /visual-grounding
{
    "query": "reset button",
    "use_indexed_components": true
}

Response:
{
    "found": true,
    "element": "Reset Button",
    "bbox": [[234, 567, 289, 612]],
    "description": "Located on the right side of the control panel",
    "image_url": "http://...",
    "confidence": "high"
}
```

**Structured captioning format:**
```json
{
    "description": "Page shows hydraulic system schematic",
    "document_type": "schematic",
    "transcribed_text": "All visible text...",
    "key_topics": ["hydraulic", "pressure", "valve"],
    "components": [
        {
            "name": "Pressure Relief Valve",
            "bbox": [[120, 340, 180, 390]],
            "type": "valve"
        }
    ]
}
```

---

## Files Modified

| File | Changes |
|------|---------|
| `config.py` | Added `RerankConfig` dataclass |
| `retrieval.py` | Integrated reranking after RRF fusion |
| `llm_client.py` | Added `caption_image_structured()`, `visual_grounding()`, `_generate_context_prefix()` |
| `ingestion.py` | Updated to use structured captioning and store component data |
| `server.py` | Added `/visual-grounding` endpoint |

## New File

| File | Purpose |
|------|---------|
| `reranking.py` | Voyage Rerank 2 client with rate limiting and graceful degradation |

---

## Architecture Flow

```
User Query
    ↓
Hybrid Search (Dense + Sparse + Combined)
    ↓
RRF Fusion (50 candidates)
    ↓
[NEW] Voyage Rerank 2 → Top 5
    ↓
LLM Generation (Qwen3-VL-Plus)
    ↓
Response + Visual Grounding (optional)
```

---

## Expected Performance Improvements

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Retrieval Accuracy | Baseline | +20-48% (reranking) |
| Context Relevance | Baseline | +67% (contextual retrieval) |
| Latency | ~2s | ~2.5s (+300-600ms reranking) |
| User Task Completion | N/A | Faster (visual grounding) |

---

## Configuration Options

### Enable/Disable Reranking
```python
# In config.py
class RerankConfig:
    enabled: bool = True  # Set to False to disable
```

### Tune Reranking Candidates
```python
candidates_to_rerank: int = 50  # Default, optimal range 30-100
top_k: int = 5  # Final results to LLM
```

### Visual Grounding at Query Time
```python
# In API request
{
    "use_indexed_components": true  # Use pre-extracted components first
}
```

---

## What Was NOT Implemented (and why)

| Technique | Evidence | Why Skipped |
|-----------|----------|-------------|
| **HyDE** | Mixed | Adds latency, hurts when LLM lacks domain knowledge |
| **GraphRAG** | Emerging | Over-engineering for document retrieval |
| **SPLADE** | Requires training | BM25 sufficient, domain fine-tuning needed |
| **Late Chunking** | +3.46% | Already using page-level embeddings |
| **BGE-M3** | Strong | Voyage multimodal-3 already best-in-class |

---

## Testing Recommendations

1. **A/B test reranking**: Compare hit rate with `enabled: true` vs `enabled: false`
2. **Measure contextual retrieval**: Track queries that now succeed with context prefix
3. **User study for visual grounding**: Measure time-to-locate for technician tasks
4. **Monitor latency**: Ensure reranking overhead is acceptable (~300-600ms)

---

## Cost Considerations

| API | Usage | Estimated Cost |
|-----|-------|----------------|
| Voyage Rerank 2 | Per query | ~$0.05/1000 requests |
| Voyage Multimodal-3 | Embedding | $0.06/1M tokens |
| Qwen3-VL | Generation + Captioning | ~$4/1M tokens |

**Total overhead from optimizations:** Minimal (~$0.05/1000 queries for reranking)
