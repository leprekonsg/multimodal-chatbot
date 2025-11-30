"""
Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
Combines dense (Voyage) and sparse (BM25) search for robust retrieval.
"""
import asyncio
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

# Use AsyncQdrantClient directly
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchParams, SparseVector

from config import config
from embeddings import voyage_embedder, sparse_embedder


@dataclass
class RetrievedDocument:
    """Document retrieved from knowledge base."""
    id: str
    score: float
    type: str
    url: Optional[str] = None
    caption: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    @property
    def content(self) -> str:
        """Get displayable content."""
        if self.type == "image":
            return self.caption or ""
        return self.text or ""
    
    @property
    def source_display(self) -> str:
        """Get source name for citation."""
        if self.title:
            return self.title
        if self.metadata and self.metadata.get("filename"):
            return self.metadata["filename"]
        return f"Source {self.id[:8]}"


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    documents: List[RetrievedDocument]
    confidence: float
    dense_count: int
    sparse_count: int
    fusion_method: str


def reciprocal_rank_fusion(
    results_dict: Dict[str, List[Tuple[str, float, dict]]],
    k: int = 60
) -> List[Tuple[str, float, dict]]:
    """
    Combine rankings from multiple retrieval systems.
    RRF Formula: score(d) = sum(1 / (k + rank_i(d)))
    """
    fused_scores = defaultdict(float)
    payloads = {}
    
    for system, doc_list in results_dict.items():
        for rank, (doc_id, score, payload) in enumerate(doc_list):
            fused_scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in payloads:
                payloads[doc_id] = payload
    
    sorted_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [
        (doc_id, score, payloads.get(doc_id, {}))
        for doc_id, score in sorted_results
    ]


class HybridRetriever:
    """
    Hybrid retrieval using AsyncQdrantClient and query_points.
    """
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(url=config.qdrant.url)
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None
    
    async def retrieve(
        self,
        query_text: str = None,
        query_image: bytes = None,
        top_k: int = None,
        filter_type: str = None
    ) -> RetrievalResult:
        top_k = top_k or config.qdrant.top_k
        
        search_filter = None
        if filter_type:
            search_filter = Filter(
                must=[FieldCondition(
                    key="type",
                    match=MatchValue(value=filter_type)
                )]
            )
        
        # Run searches in parallel
        dense_task = self._dense_search(
            query_text=query_text,
            query_image=query_image,
            limit=top_k * 2,
            filter=search_filter
        )
        
        if query_text:
            sparse_task = self._sparse_search(
                query_text=query_text,
                limit=top_k * 2,
                filter=search_filter
            )
            dense_results, sparse_results = await asyncio.gather(
                dense_task, sparse_task
            )
        else:
            dense_results = await dense_task
            sparse_results = []
        
        # Fuse results
        results_dict = {"dense": dense_results}
        if sparse_results:
            results_dict["sparse"] = sparse_results
        
        fused = reciprocal_rank_fusion(results_dict)
        top_results = fused[:top_k]
        
        confidence = self._calculate_confidence(dense_results, sparse_results, top_results)
        
        documents = [
            self._payload_to_document(doc_id, score, payload)
            for doc_id, score, payload in top_results
        ]
        
        dense_ids = {r[0] for r in dense_results[:top_k]}
        sparse_ids = {r[0] for r in sparse_results[:top_k]} if sparse_results else set()
        
        return RetrievalResult(
            documents=documents,
            confidence=confidence,
            dense_count=len([d for d in documents if d.id in dense_ids]),
            sparse_count=len([d for d in documents if d.id in sparse_ids]),
            fusion_method="RRF"
        )
    
    async def _dense_search(
        self,
        query_text: str = None,
        query_image: bytes = None,
        limit: int = 10,
        filter = None
    ) -> List[Tuple[str, float, dict]]:
        
        query_vector = await voyage_embedder.encode_query(
            text=query_text,
            image=query_image
        )
        query_vector = query_vector.tolist()
        
        # FIX: Use query_points instead of search
        response = await self.client.query_points(
            collection_name=config.qdrant.collection_name,
            query=query_vector,
            using=config.qdrant.dense_vector_name,
            limit=limit,
            query_filter=filter,  # Changed from filter=filter
            search_params=SearchParams(hnsw_ef=config.qdrant.hnsw_ef)
        )
        
        return [(str(r.id), r.score, r.payload) for r in response.points]
    
    async def _sparse_search(
        self,
        query_text: str,
        limit: int = 10,
        filter = None
    ) -> List[Tuple[str, float, dict]]:
        
        if not query_text:
            return []
        
        sparse_query = sparse_embedder.encode(query_text)
        if not sparse_query.get("indices"):
            return []
        
        # FIX: Use query_points with SparseVector
        response = await self.client.query_points(
            collection_name=config.qdrant.collection_name,
            query=SparseVector(
                indices=sparse_query["indices"],
                values=sparse_query["values"]
            ),
            using=config.qdrant.sparse_vector_name,
            limit=limit,
            query_filter=filter  # Changed from filter=filter
        )
        
        return [(str(r.id), r.score, r.payload) for r in response.points]
    
    def _calculate_confidence(self, dense, sparse, fused) -> float:
        if not fused: return 0.0
        top_score = dense[0][1] if dense else 0.0
        gap = (dense[0][1] - dense[1][1]) if len(dense) >= 2 else 0.2
        
        agreement = 0.0
        if dense and sparse:
            d_set = {r[0] for r in dense[:3]}
            s_set = {r[0] for r in sparse[:3]}
            agreement = len(d_set & s_set) * 0.1
            
        return min(1.0, max(0.0, top_score * 0.5 + min(gap * 2, 0.3) + agreement))
    
    def _payload_to_document(self, doc_id, score, payload) -> RetrievedDocument:
        return RetrievedDocument(
            id=doc_id,
            score=score,
            type=payload.get("type", "unknown"),
            url=payload.get("url"),
            caption=payload.get("caption"),
            text=payload.get("text"),
            title=payload.get("title"),
            metadata={k: v for k, v in payload.items() if k not in ["type", "url", "caption", "text", "title"]}
        )


# Singleton instance
retriever = HybridRetriever()