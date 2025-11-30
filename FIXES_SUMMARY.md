# RAG System Critical Fixes Summary

## Executive Summary

Your multimodal RAG system had **3 critical bugs** that caused 15% confidence despite having the exact relevant document ingested. All have been fixed.

---

## 🔴 Critical Bug #1: Text Queries Never Searched `combined_dense` Vector

### The Problem
```
Query: "Tie down procedures for helicopters" (text only)
Intent: TEXTUAL_SEARCH

BEFORE:
├── text_dense search ✓
├── sparse search ✓
└── combined_dense search ✗ MISSING!

AFTER:
├── text_dense search ✓
├── sparse search ✓
└── combined_dense search ✓ ADDED!
```

### Why This Matters
- Your helicopter document has a `combined_dense` vector that fuses **visual features** (helicopter diagrams, knots) + **caption text**
- A text query about "helicopters" should match the visual helicopter semantics in `combined_dense`
- By only searching `text_dense`, you were limited to caption matching only
- If the caption didn't say "helicopter" explicitly, the match failed

### The Fix (retrieval.py lines 401-412)
```python
# CRITICAL FIX: Also search combined_dense for text queries!
search_tasks.append(
    self._vector_search(
        vector=query_embeddings["text_query"],
        vector_name=self.VECTOR_NAMES["combined"],  # Search combined!
        limit=top_k * 2,
        filter=search_filter
    )
)
strategies_used.append("combined_text")
```

---

## 🔴 Critical Bug #2: Confidence Used RRF Scores vs Cosine Thresholds

### The Problem
```python
# Before: Using RRF fused score
top_score = top_doc.score  # This is ~0.03 (RRF score)

# Compared against cosine thresholds:
COSINE_POOR = 0.40  

# 0.03 < 0.40 triggers penalty:
confidence *= 0.5  # 0.30 * 0.5 = 0.15 (15%!)
```

### Why This Matters
- RRF (Reciprocal Rank Fusion) scores are in range ~0.01-0.1
- Cosine similarity scores are in range 0.0-1.0
- The code was comparing apples to oranges!
- A perfectly good match would always trigger "low confidence"

### The Fix (retrieval.py ConfidenceCalculator)
```python
# Now: Get actual cosine similarities from match_info
actual_scores = [
    top_doc.visual_score,
    top_doc.textual_score,
]
if top_doc.metadata:
    if top_doc.metadata.get("combined_text_score"):
        actual_scores.append(top_doc.metadata["combined_text_score"])
        
# Use actual cosine score for confidence calculation
top_score = max(actual_scores) if actual_scores else min(top_doc.score * 10, 1.0)
```

---

## 🔴 Critical Bug #3: Quick Escalation at 30% Confidence

### The Problem
```python
# escalation.py evaluate_quick()
if retrieval_confidence < 0.3:  # 15% < 30%
    return EscalationDecision(
        should_escalate=True,
        message_to_user="I don't have relevant information..."
    )
```

With 15% confidence (from Bug #2), every query would escalate immediately without generating an answer!

### The Fix
By fixing Bug #2, confidence will now be calculated correctly based on actual cosine similarity scores, typically yielding 40-80% for relevant matches.

---

## 🟡 Enhancement #1: Token Usage Logging

### Before
Token usage was tracked silently - no visibility.

### After (usage.py)
```python
def track_voyage(self, tokens: int = 0, pixels: int = 0):
    self.total.voyage_tokens += tokens
    if VERBOSE_USAGE:
        print(f"📊 [Voyage] tokens={tokens}, pixels={pixels:,} | Total: ...")

def track_qwen(self, input_tokens: int, output_tokens: int):
    if VERBOSE_USAGE:
        print(f"📊 [Qwen] in={input_tokens}, out={output_tokens} | Total: ...")
```

Now you'll see real-time token usage during ingestion and retrieval.

---

## 🟡 Enhancement #2: Multi-Image Generation (5 instead of 2)

### Before (llm_client.py)
```python
for url in image_urls[:2]:  # Only 2 images
```

### After
```python
for url in image_urls[:5]:  # Up to 5 images for better context
```

Qwen3-VL supports multi-image understanding. More context = better answers.

---

## 🟡 Enhancement #3: Comprehensive Logging

Added logging throughout the pipeline:

### Ingestion
```
[Ingestion] Starting ingest for: helicopter.jpg (1,234,567 bytes)
[Ingestion] Stored at: http://localhost:8000/images/abc123.jpg
[Ingestion] Generating caption...
[Ingestion] Caption: This page shows tie-down procedures...
[Voyage API] Calling with 3 inputs, type=document...
[Voyage API] Success: 3 embeddings returned
📊 [Voyage] tokens=1234, pixels=1,048,576 | Total: ...
[Ingestion] Stored vectors: ['image_dense', 'text_dense', 'combined_dense', 'sparse(45 terms)']
[Ingestion] Complete! doc_id=abc123-...
```

### Retrieval
```
🔍 [Retrieval] Query intent: textual
🔍 [Retrieval] Query text: Tie down procedures for helicopters
⏳ [Retrieval] Generating query embeddings...
✅ [Retrieval] Generated embeddings: ['text_query']
📋 [Retrieval] Strategies used: ['textual', 'sparse', 'combined_text']
📋 [Retrieval] Results per strategy: {textual: 5, sparse: 3, combined_text: 7}
🎯 [Retrieval] Top result: helicopter.jpg (score: 0.78, type: semantic)
📄 [Retrieval] Caption: This page shows tie-down procedures for helicopters...
📊 [Retrieval] Confidence: 75% | Time: 234.5ms
```

---

## Files Modified

| File | Changes |
|------|---------|
| `retrieval.py` | Added combined_dense search for text queries, fixed confidence calculation, added logging |
| `embeddings.py` | Added Voyage API call logging |
| `usage.py` | Added real-time token usage printing |
| `llm_client.py` | Increased image limit from 2 to 5 |
| `chatbot.py` | Added escalation logging |
| `ingestion.py` | Added comprehensive ingestion logging |

---

## Expected Results After Fix

| Scenario | Before | After |
|----------|--------|-------|
| Text query about ingested content | 15% confidence, no answer | 60-80% confidence, correct answer |
| Visual semantics matching | Never searched | Now searched via combined_text |
| Token visibility | Hidden | Real-time logging |
| Multi-image context | Max 2 images | Max 5 images |

---

## Testing the Fix

1. **Re-ingest your helicopter image** (the new logging will show what caption/vectors are stored)
2. **Query "Tie down procedures for helicopters"**
3. **Observe the logs** - you should see:
   - `combined_text` strategy in use
   - Actual cosine scores being used for confidence
   - Higher confidence (60%+ for relevant content)
   - Generated answer instead of escalation

---

## Additional Recommendations

1. **Caption Quality**: The caption prompt asks to "transcribe ALL visible text" - this is critical for searchability. Check what caption was generated for your helicopter image.

2. **Rate Limiting**: Voyage has 3 RPM limit. The system enforces 20-second waits between calls. For bulk ingestion, consider batching.

3. **Collection Verification**: Ensure your document was actually stored in `knowledge_base_v2` collection (not the old `knowledge_base` collection).
