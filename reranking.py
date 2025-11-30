"""
Reranking Module
Cross-encoder reranking using Voyage Rerank 2 for precision improvement.

Research Evidence:
- 20-48% accuracy improvement (Databricks research)
- Hit rate increase from 58% to 87% in production (Coalfire case study)
- ~595ms latency overhead, justified by quality gains

Architecture:
- Retrieve 50 candidates via hybrid search
- Rerank with cross-encoder
- Return top_k (default 5) to LLM context

This module uses the same API key as Voyage embeddings, minimizing overhead.
"""
import asyncio
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import httpx

from config import config
from usage import tracker


@dataclass
class RerankResult:
    """Result from reranking operation."""
    doc_id: str
    original_rank: int
    rerank_score: float
    payload: Dict[str, Any]
    match_info: Dict[str, Any]


class VoyageReranker:
    """
    Voyage Rerank 2 client for cross-encoder reranking.
    
    Key benefits over bi-encoder:
    - Captures deep query-document interactions
    - Significantly improves precision on technical queries
    - Particularly effective for error codes, part numbers, procedures
    
    Rate Limiting:
    - Uses same rate limiter as embeddings (3 RPM)
    - In practice, reranking is called once per query, not per document
    """
    
    def __init__(self):
        self.api_key = config.voyage.api_key
        self.model = config.rerank.model_name
        self.api_url = "https://api.voyageai.com/v1/rerank"
        self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,  # Reranking is fast, 30s is generous
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def rerank(
        self,
        query: str,
        documents: List[Tuple[str, float, Dict, Dict]],
        top_k: int = None
    ) -> List[Tuple[str, float, Dict, Dict]]:
        """
        Rerank documents using Voyage Rerank 2 cross-encoder.
        
        Args:
            query: User's search query
            documents: List of (doc_id, rrf_score, payload, match_info) from hybrid search
            top_k: Number of top results to return (default from config)
        
        Returns:
            Reranked list of (doc_id, rerank_score, payload, match_info)
        
        Note: If reranking fails, returns original order (graceful degradation)
        """
        if not config.rerank.enabled:
            return documents[:top_k] if top_k else documents
        
        if not documents:
            return []
        
        top_k = top_k or config.rerank.top_k
        
        # Limit candidates to rerank (research suggests 50-100 is optimal)
        candidates = documents[:config.rerank.candidates_to_rerank]
        
        # Extract text content for reranking
        # For technical manuals, we use caption + extracted_text for images
        doc_texts = []
        for doc_id, score, payload, match_info in candidates:
            if payload.get("type") == "image":
                # Combine caption and extracted text for image documents
                text = payload.get("caption", "")
                if payload.get("extracted_text"):
                    text += f"\n{payload['extracted_text']}"
            else:
                text = payload.get("text", "") or payload.get("caption", "")
            
            # Truncate to avoid API limits (max ~8000 chars per doc)
            doc_texts.append(text[:8000] if text else "")
        
        # Skip reranking if no valid texts
        if not any(doc_texts):
            print("⚠️ [Rerank] No valid text content to rerank, returning original order")
            return documents[:top_k]
        
        start_time = time.time()
        
        try:
            payload = {
                "model": self.model,
                "query": query,
                "documents": doc_texts,
                "top_k": min(top_k, len(candidates)),
                "return_documents": False  # We already have the documents
            }
            
            response = await self.client.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Track usage (Voyage rerank has its own token counting)
            if data.get("usage"):
                tracker.track_voyage(tokens=data["usage"].get("total_tokens", 0))
            
            # Reorder documents based on rerank scores
            reranked_indices = data.get("data", [])
            
            if not reranked_indices:
                print("⚠️ [Rerank] Empty response, returning original order")
                return documents[:top_k]
            
            # Build reranked list
            reranked = []
            for item in reranked_indices:
                idx = item["index"]
                rerank_score = item["relevance_score"]
                
                if idx < len(candidates):
                    doc_id, orig_score, payload, match_info = candidates[idx]
                    # Add rerank score to match_info for downstream use
                    match_info["rerank_score"] = rerank_score
                    match_info["original_rrf_score"] = orig_score
                    reranked.append((doc_id, rerank_score, payload, match_info))
            
            print(f"✅ [Rerank] Reranked {len(candidates)} candidates in {latency_ms:.1f}ms")
            
            # Log significant reordering (useful for debugging)
            if reranked and len(candidates) > 1:
                top_before = candidates[0][0][:8]
                top_after = reranked[0][0][:8]
                if top_before != top_after:
                    print(f"📊 [Rerank] Top result changed: {top_before}... → {top_after}...")
            
            return reranked[:top_k]
            
        except httpx.HTTPStatusError as e:
            print(f"❌ [Rerank] API error {e.response.status_code}: {e.response.text[:200]}")
            # Graceful degradation: return original order
            return documents[:top_k]
            
        except Exception as e:
            print(f"❌ [Rerank] Failed: {e}")
            # Graceful degradation: return original order
            return documents[:top_k]
    
    async def rerank_for_visual_grounding(
        self,
        query: str,
        documents: List[Tuple[str, float, Dict, Dict]],
        top_k: int = 3
    ) -> List[Tuple[str, float, Dict, Dict]]:
        """
        Specialized reranking for visual grounding queries.
        
        For queries like "where is the reset button?", we want to:
        1. Rerank to find the most relevant page
        2. Prioritize pages with component annotations in metadata
        
        Args:
            query: User's localization query
            documents: Candidates from hybrid search
            top_k: Number to return (default 3 for visual grounding)
        
        Returns:
            Top candidates for visual grounding
        """
        # First, standard reranking
        reranked = await self.rerank(query, documents, top_k=top_k * 2)
        
        # Post-filter: boost documents with component data if available
        boosted = []
        for doc_id, score, payload, match_info in reranked:
            # Check if document has component bounding box data
            components = payload.get("components", [])
            if components:
                # Boost score slightly for documents with structured component data
                score *= 1.1
            boosted.append((doc_id, score, payload, match_info))
        
        # Re-sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)
        
        return boosted[:top_k]


# Singleton instance
reranker = VoyageReranker()
