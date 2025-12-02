"""
Enhanced Hybrid Retrieval System

Key Improvements:
1. Multi-strategy search (visual, textual, combined)
2. Perceptual hash exact matching (fast path)
3. Calibrated confidence scores
4. Query understanding for optimal strategy selection
5. Re-ranking stage for final results
6. ASCII logging (no emoji encoding issues)
"""
import asyncio
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, SearchParams, 
    SparseVector, NamedVector, Prefetch, Query, FusionQuery
)

from config import config
from embeddings import (
    voyage_embedder_v2,
    sparse_embedder_v2,
    ImageFingerprint,
    PerceptualHasher
)
from reranking import reranker


class QueryIntent(Enum):
    """Classified query intent for strategy selection."""
    VISUAL_SEARCH = "visual"       # "What is this?" with image
    TEXTUAL_SEARCH = "textual"     # Pure text query
    MULTIMODAL = "multimodal"      # Text + image query
    EXACT_MATCH = "exact"          # Find this exact image
    SIMILARITY = "similarity"       # Find similar images


@dataclass
class RetrievedDocumentV2:
    """Enhanced retrieved document with match details."""
    id: str
    score: float
    type: str
    url: Optional[str] = None
    caption: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    # Match details
    match_type: str = "semantic"  # "exact", "visual", "textual", "semantic"
    visual_score: float = 0.0
    textual_score: float = 0.0
    fingerprint_distance: int = 64  # Hamming distance (64 = no match)
    
    @property
    def content(self) -> str:
        if self.type == "image":
            return self.caption or ""
        return self.text or ""
    
    @property
    def source_display(self) -> str:
        if self.title:
            return self.title
        if self.metadata and self.metadata.get("filename"):
            return self.metadata["filename"]
        return f"Source {self.id[:8]}"
    
    @property
    def is_exact_match(self) -> bool:
        return self.fingerprint_distance < 5


@dataclass
class RetrievalResultV2:
    """Enhanced retrieval result with detailed metrics."""
    documents: List[RetrievedDocumentV2]
    confidence: float
    query_intent: QueryIntent
    
    # Detailed metrics
    exact_matches: int = 0
    visual_matches: int = 0
    textual_matches: int = 0
    semantic_matches: int = 0
    
    # Strategy used
    strategies_used: List[str] = field(default_factory=list)
    
    # Timing
    retrieval_ms: float = 0.0


class QueryClassifier:
    """
    Classifies query intent to select optimal retrieval strategy.
    """
    
    VISUAL_TRIGGERS = [
        "what is this", "identify", "recognize", "show me similar",
        "find this", "what does this show", "describe this image",
        "what's in this", "what am i looking at"
    ]
    
    EXACT_TRIGGERS = [
        "find exact", "same image", "this exact", "duplicate",
        "where is this from", "original"
    ]
    
    @classmethod
    def classify(
        cls,
        query_text: Optional[str],
        has_image: bool
    ) -> QueryIntent:
        """Determine query intent based on input."""
        
        if not query_text and has_image:
            # Pure image upload - likely visual search or exact match
            return QueryIntent.VISUAL_SEARCH
        
        if query_text and not has_image:
            return QueryIntent.TEXTUAL_SEARCH
        
        if query_text and has_image:
            query_lower = query_text.lower()
            
            # Check for exact match intent
            for trigger in cls.EXACT_TRIGGERS:
                if trigger in query_lower:
                    return QueryIntent.EXACT_MATCH
            
            # Check for visual search intent
            for trigger in cls.VISUAL_TRIGGERS:
                if trigger in query_lower:
                    return QueryIntent.VISUAL_SEARCH
            
            # Default to multimodal
            return QueryIntent.MULTIMODAL
        
        return QueryIntent.TEXTUAL_SEARCH


class ConfidenceCalculator:
    """
    Calibrated confidence scoring based on retrieval quality signals.
    
    IMPORTANT: Uses the actual cosine similarity scores, NOT the RRF fused scores.
    RRF scores are much smaller (0.01-0.1) and not comparable to cosine thresholds.
    
    Note: Sparse/BM25 scores are NOT cosine similarities and must be excluded.
    """
    
    # Empirically calibrated thresholds for Voyage embeddings (cosine similarity)
    COSINE_EXCELLENT = 0.85  # Very strong match
    COSINE_GOOD = 0.70       # Good match
    COSINE_MARGINAL = 0.55   # Weak match
    COSINE_POOR = 0.40       # Unlikely relevant
    
    # Cosine-based strategy suffixes (exclude sparse which uses BM25 scoring)
    COSINE_SCORE_KEYS = ["visual_score", "textual_score", "combined_score", "combined_text_score"]
    
    @classmethod
    def _collect_cosine_scores(cls, doc: RetrievedDocumentV2) -> Dict[str, float]:
        """
        Collect all cosine similarity scores from a document.
        Excludes sparse scores as they're BM25 weights, not cosine similarities.
        
        Returns: Dict mapping score_name -> score_value (only non-zero scores)
        """
        scores = {}
        
        # Direct attributes
        if doc.visual_score > 0:
            scores["visual_score"] = doc.visual_score
        if doc.textual_score > 0:
            scores["textual_score"] = doc.textual_score
        
        # Metadata scores (from all strategies)
        if doc.metadata:
            for key in cls.COSINE_SCORE_KEYS:
                value = doc.metadata.get(key)
                if value and value > 0:
                    scores[key] = value
        
        return scores
    
    @classmethod
    def calculate(
        cls,
        documents: List[RetrievedDocumentV2],
        query_intent: QueryIntent,
        verbose: bool = False
    ) -> float:
        """
        Calculate calibrated confidence score.
        
        Signals used:
        1. Top match score (most important) - uses ACTUAL cosine similarity
        2. Score gap between top results (distinctiveness)
        3. Match type alignment with query intent
        4. Exact match bonus
        
        Args:
            documents: Retrieved documents (best first)
            query_intent: Classified intent of the query
            verbose: If True, print diagnostic info
        """
        if not documents:
            return 0.0
        
        top_doc = documents[0]
        
        # Collect all cosine scores for top document
        top_scores = cls._collect_cosine_scores(top_doc)
        
        # Get the best actual cosine similarity score
        if top_scores:
            top_score = max(top_scores.values())
            top_score_source = max(top_scores.items(), key=lambda x: x[1])[0]
        else:
            # Fallback: RRF scores are typically 0.01-0.1, so scale them up
            # This is a rough heuristic for when actual scores aren't available
            top_score = min(top_doc.score * 10, 1.0)
            top_score_source = "rrf_scaled"
        
        # 1. Base confidence from top score
        if top_score >= cls.COSINE_EXCELLENT:
            base_conf = 0.85
        elif top_score >= cls.COSINE_GOOD:
            base_conf = 0.70
        elif top_score >= cls.COSINE_MARGINAL:
            base_conf = 0.50
        else:
            base_conf = 0.35  # Slightly higher floor for better UX
        
        # 2. Score gap bonus (distinct answer is more confident)
        gap_bonus = 0.0
        doc2_max = 0.0
        if len(documents) >= 2:
            doc2_scores = cls._collect_cosine_scores(documents[1])
            doc2_max = max(doc2_scores.values()) if doc2_scores else 0
            
            gap = top_score - doc2_max
            if gap > 0.15:
                gap_bonus = 0.08  # Very distinct
            elif gap > 0.08:
                gap_bonus = 0.04  # Somewhat distinct
        else:
            # Only one document - give a small bonus for distinctiveness
            gap_bonus = 0.05
        
        # 3. Intent alignment bonus
        intent_bonus = 0.0
        if query_intent == QueryIntent.VISUAL_SEARCH:
            visual_relevant = max(
                top_scores.get("visual_score", 0),
                top_scores.get("combined_score", 0)
            )
            if visual_relevant > 0.7:
                intent_bonus = 0.05
        elif query_intent == QueryIntent.TEXTUAL_SEARCH:
            text_relevant = max(
                top_scores.get("textual_score", 0),
                top_scores.get("combined_text_score", 0),
                top_scores.get("combined_score", 0)
            )
            if text_relevant > 0.7:
                intent_bonus = 0.05
        elif query_intent == QueryIntent.MULTIMODAL:
            if top_scores.get("combined_score", 0) > 0.7:
                intent_bonus = 0.05
        
        # 4. Exact match bonus (perceptual hash match)
        exact_bonus = 0.0
        if top_doc.is_exact_match:
            exact_bonus = 0.15  # Strong boost for exact matches
        
        # Calculate final confidence (cap at 0.95 unless exact match)
        confidence = base_conf + gap_bonus + intent_bonus + exact_bonus
        
        # Exact matches can go to 1.0, others cap at 0.95
        max_conf = 1.0 if top_doc.is_exact_match else 0.95
        confidence = min(max_conf, confidence)
        
        # Apply penalty for very low top scores
        if top_score < cls.COSINE_POOR:
            confidence *= 0.7  # Less aggressive penalty
        
        # Diagnostic logging (ASCII only)
        if verbose:
            print(f"[Confidence] Top scores: {top_scores}")
            print(f"[Confidence] Best score: {top_score:.4f} from {top_score_source}")
            print(f"[Confidence] Base: {base_conf:.2f} | Gap: +{gap_bonus:.2f} (doc2_max={doc2_max:.4f}) | Intent: +{intent_bonus:.2f} | Exact: +{exact_bonus:.2f}")
            print(f"[Confidence] Final: {confidence:.2%}")
        
        return confidence


class HybridRetrieverV2:
    """
    Enhanced hybrid retrieval with multiple search strategies.
    
    Strategies:
    1. Exact Match: Perceptual hash lookup (instant)
    2. Visual Search: image_dense vector
    3. Textual Search: text_dense vector + sparse
    4. Semantic Search: combined_dense vector
    5. RRF Fusion: Combine all strategies
    """
    
    VECTOR_NAMES = {
        "image": "image_dense",
        "text": "text_dense",
        "combined": "combined_dense",
        "sparse": "sparse"
    }
    
    def __init__(self):
        self._client = None
        self.classifier = QueryClassifier()
        self.confidence_calc = ConfidenceCalculator()
        self.hasher = PerceptualHasher()
    
    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(url=config.qdrant.url)
        return self._client
    
    @property
    def collection_name(self) -> str:
        return f"{config.qdrant.collection_name}_v2"
    
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
    ) -> RetrievalResultV2:
        """
        Main retrieval endpoint with automatic strategy selection.
        """
        import time
        start_time = time.time()
        
        top_k = top_k or config.qdrant.top_k
        
        # Classify query intent
        query_intent = self.classifier.classify(query_text, query_image is not None)
        print(f"[Retrieval] Query intent: {query_intent.value}")
        print(f"[Retrieval] Query text: {query_text[:100] if query_text else 'None'}...")
        print(f"[Retrieval] Has image: {query_image is not None}")
        
        # Build filter
        search_filter = None
        if filter_type:
            search_filter = Filter(
                must=[FieldCondition(
                    key="type",
                    match=MatchValue(value=filter_type)
                )]
            )
        
        # Generate query embeddings
        print(f"[Retrieval] Generating query embeddings...")
        query_embeddings = await voyage_embedder_v2.encode_query_adaptive(
            text=query_text,
            image=query_image,
            query_intent=query_intent.value
        )
        print(f"[Retrieval] Generated embeddings: {list(query_embeddings.keys())}")

        # ===== PARALLEL SEARCH OPTIMIZATION =====
        # Execute ALL search strategies concurrently (exact hash + vector searches)
        # This eliminates blocking when exact hash fails (common case)

        search_tasks = {}

        # Strategy 1: Exact match via perceptual hash (if image provided)
        if "fingerprint" in query_embeddings:
            search_tasks["exact"] = self._exact_hash_search(
                query_embeddings["fingerprint"],
                limit=top_k,
                filter=search_filter
            )

        # Strategy 2: Visual search (if image query available)
        if "image_query" in query_embeddings:
            search_tasks["visual"] = self._vector_search(
                vector=query_embeddings["image_query"],
                vector_name=self.VECTOR_NAMES["image"],
                limit=top_k * 2,
                filter=search_filter
            )

        # Strategy 3: Text search (if text query available)
        if "text_query" in query_embeddings:
            search_tasks["textual"] = self._vector_search(
                vector=query_embeddings["text_query"],
                vector_name=self.VECTOR_NAMES["text"],
                limit=top_k * 2,
                filter=search_filter
            )

            # Strategy 3b: Sparse search
            search_tasks["sparse"] = self._sparse_search(
                query_text=query_text,
                limit=top_k * 2,
                filter=search_filter
            )

            # Strategy 3c: Combined dense for text queries
            search_tasks["combined_text"] = self._vector_search(
                vector=query_embeddings["text_query"],
                vector_name=self.VECTOR_NAMES["combined"],
                limit=top_k * 2,
                filter=search_filter
            )

        # Strategy 4: Combined search (if multimodal query - uses combined embedding)
        if "combined_query" in query_embeddings:
            search_tasks["combined"] = self._vector_search(
                vector=query_embeddings["combined_query"],
                vector_name=self.VECTOR_NAMES["combined"],
                limit=top_k * 2,
                filter=search_filter
            )

        # Execute all searches in parallel
        print(f"[Retrieval] Executing {len(search_tasks)} search strategies in parallel...")
        search_results = await asyncio.gather(*search_tasks.values(), return_exceptions=True)

        # Build results dict, handling exceptions
        results_dict = {}
        strategies_used = []
        for strategy, result in zip(search_tasks.keys(), search_results):
            if isinstance(result, Exception):
                print(f"[!] Search strategy '{strategy}' failed: {result}")
                continue
            if result:
                results_dict[strategy] = result
                strategies_used.append(strategy)

        # Check for high-confidence exact matches (early return optimization)
        exact_results = results_dict.get("exact", [])
        if exact_results and exact_results[0][2] < 5:  # Hamming distance < 5
            docs = [self._to_document(r, match_type="exact") for r in exact_results[:top_k]]
            print(f"[Retrieval] Early return: Found exact match with distance < 5")

            return RetrievalResultV2(
                documents=docs,
                confidence=0.98,  # Very high confidence for exact match
                query_intent=query_intent,
                exact_matches=len(docs),
                strategies_used=["exact_hash"],
                retrieval_ms=(time.time() - start_time) * 1000
            )
        
        # Fuse results using weighted RRF
        fused = self._weighted_rrf(results_dict, query_intent)
        
        # Log search results
        print(f"[Retrieval] Strategies used: {strategies_used}")
        results_summary = ", ".join(f"{k}: {len(v)}" for k, v in results_dict.items())
        print(f"[Retrieval] Results per strategy: {{{results_summary}}}")
        
        # === RERANKING STAGE (20-48% accuracy improvement) ===
        if query_text and len(fused) > 1:
            try:
                print(f"[Retrieval] Reranking {min(len(fused), config.rerank.candidates_to_rerank)} candidates...")
                fused = await reranker.rerank(
                    query=query_text,
                    documents=fused,
                    top_k=top_k
                )
            except Exception as e:
                print(f"[!] Reranking failed, using RRF order: {e}")
        
        top_results = fused[:top_k]
        
        # Convert to documents with match details
        documents = []
        visual_matches = textual_matches = semantic_matches = exact_matches = 0
        
        for doc_id, score, payload, match_info in top_results:
            doc = self._to_document_with_info(
                doc_id, score, payload, match_info
            )
            documents.append(doc)
            
            # Count match types
            if doc.is_exact_match:
                exact_matches += 1
            elif doc.match_type == "visual":
                visual_matches += 1
            elif doc.match_type == "textual":
                textual_matches += 1
            else:
                semantic_matches += 1
        
        # Calculate calibrated confidence (with verbose logging if enabled)
        import os
        verbose_confidence = os.environ.get("VERBOSE_USAGE", "1") == "1"
        confidence = self.confidence_calc.calculate(documents, query_intent, verbose=verbose_confidence)
        
        retrieval_ms = (time.time() - start_time) * 1000
        
        # Log results summary (ASCII only)
        if documents:
            top_doc = documents[0]
            print(f"[Retrieval] Top result: {top_doc.source_display} (score: {top_doc.score:.4f}, type: {top_doc.match_type})")
            caption_preview = (top_doc.caption or 'None')[:150]
            print(f"[Retrieval] Caption: {caption_preview}...")
            
            # NEW: Print components for top result to debug visual grounding
            if top_doc.metadata and top_doc.metadata.get("components"):
                comps = top_doc.metadata["components"]
                print(f"[Retrieval] Indexed Components ({len(comps)}):")
                for c in comps[:5]:  # Show first 5
                    print(f"   - {c.get('label', 'Unknown')}: {c.get('bbox_2d', 'No bbox')}")
                if len(comps) > 5:
                    print(f"   ... and {len(comps)-5} more")
            else:
                print("[Retrieval] No components indexed for this document")
        else:
            print(f"[!] No documents found!")
        print(f"[Retrieval] Confidence: {confidence:.2%} | Time: {retrieval_ms:.1f}ms")
        
        return RetrievalResultV2(
            documents=documents,
            confidence=confidence,
            query_intent=query_intent,
            exact_matches=exact_matches,
            visual_matches=visual_matches,
            textual_matches=textual_matches,
            semantic_matches=semantic_matches,
            strategies_used=strategies_used,
            retrieval_ms=retrieval_ms
        )
    
    async def _exact_hash_search(
        self,
        query_fingerprint: ImageFingerprint,
        limit: int,
        filter = None
    ) -> List[Tuple[str, float, int, dict]]:
        """
        Search for exact/near-duplicate images using perceptual hash.
        Returns: List of (doc_id, score, hamming_distance, payload)
        """
        try:
            # Get candidates with fingerprints
            prefix = query_fingerprint.phash[:16]
            
            results = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value="image")
                        )
                    ]
                ),
                limit=1000,
                with_payload=True
            )
            
            # Compute hamming distances
            matches = []
            for point in results[0]:
                if point.payload.get("fingerprint"):
                    stored_fp = ImageFingerprint.from_dict(point.payload["fingerprint"])
                    distance = query_fingerprint.hamming_distance(stored_fp)
                    
                    # Only include near-matches (distance < 20)
                    if distance < 20:
                        similarity = 1.0 - (distance / 64.0)
                        matches.append((
                            str(point.id),
                            similarity,
                            distance,
                            point.payload
                        ))
            
            # Sort by distance (ascending)
            matches.sort(key=lambda x: x[2])
            return matches[:limit]
            
        except Exception as e:
            print(f"[!] Exact hash search error: {e}")
            return []
    
    async def _vector_search(
        self,
        vector: Any,
        vector_name: str,
        limit: int,
        filter = None
    ) -> List[Tuple[str, float, dict]]:
        """Execute single vector search."""
        try:
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=vector.tolist() if hasattr(vector, 'tolist') else vector,
                using=vector_name,
                limit=limit,
                query_filter=filter,
                search_params=SearchParams(hnsw_ef=config.qdrant.hnsw_ef)
            )
            
            return [(str(r.id), r.score, r.payload) for r in response.points]
            
        except Exception as e:
            print(f"[!] Vector search error ({vector_name}): {e}")
            return []
    
    async def _sparse_search(
        self,
        query_text: str,
        limit: int,
        filter = None
    ) -> List[Tuple[str, float, dict]]:
        """Execute sparse (BM25) search."""
        if not query_text:
            return []
        
        try:
            sparse_query = sparse_embedder_v2.encode(query_text)
            if not sparse_query.get("indices"):
                return []
            
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=SparseVector(
                    indices=sparse_query["indices"],
                    values=sparse_query["values"]
                ),
                using=self.VECTOR_NAMES["sparse"],
                limit=limit,
                query_filter=filter
            )
            
            return [(str(r.id), r.score, r.payload) for r in response.points]
            
        except Exception as e:
            print(f"[!] Sparse search error: {e}")
            return []
    
    def _weighted_rrf(
        self,
        results_dict: Dict[str, List],
        query_intent: QueryIntent,
        k: int = 60
    ) -> List[Tuple[str, float, dict, dict]]:
        """
        Weighted Reciprocal Rank Fusion.
        
        Weights strategies based on query intent.
        """
        # Define strategy weights based on intent
        weights = {
            QueryIntent.VISUAL_SEARCH: {
                "exact": 3.0, "visual": 2.0, "combined": 1.5, "combined_text": 1.2,
                "textual": 0.5, "sparse": 0.3
            },
            QueryIntent.TEXTUAL_SEARCH: {
                "exact": 1.0, "visual": 0.3, "combined": 1.5, "combined_text": 1.5,
                "textual": 2.0, "sparse": 1.5
            },
            QueryIntent.MULTIMODAL: {
                "exact": 2.0, "visual": 1.2, "combined": 1.5, "combined_text": 1.0,
                "textual": 1.2, "sparse": 1.0
            },
            QueryIntent.EXACT_MATCH: {
                "exact": 5.0, "visual": 2.0, "combined": 1.0, "combined_text": 0.8,
                "textual": 0.3, "sparse": 0.2
            },
            QueryIntent.SIMILARITY: {
                "exact": 1.0, "visual": 2.5, "combined": 1.5, "combined_text": 1.0,
                "textual": 0.5, "sparse": 0.3
            }
        }
        
        strategy_weights = weights.get(query_intent, weights[QueryIntent.MULTIMODAL])
        
        fused_scores = defaultdict(float)
        payloads = {}
        match_info = defaultdict(dict)
        
        for strategy, doc_list in results_dict.items():
            weight = strategy_weights.get(strategy, 1.0)
            
            for rank, item in enumerate(doc_list):
                # Handle different result formats
                if len(item) == 4:  # Exact match format
                    doc_id, score, distance, payload = item
                    match_info[doc_id]["hamming_distance"] = distance
                else:
                    doc_id, score, payload = item
                
                # Weighted RRF score
                fused_scores[doc_id] += weight * (1.0 / (k + rank))
                
                if doc_id not in payloads:
                    payloads[doc_id] = payload
                
                # Track which strategies found this doc
                if strategy not in match_info[doc_id].get("strategies", []):
                    match_info[doc_id].setdefault("strategies", []).append(strategy)
                match_info[doc_id][f"{strategy}_score"] = score
        
        # Sort by fused score
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            (doc_id, score, payloads.get(doc_id, {}), match_info.get(doc_id, {}))
            for doc_id, score in sorted_results
        ]
    
    def _to_document(
        self,
        result: Tuple,
        match_type: str = "semantic"
    ) -> RetrievedDocumentV2:
        """Convert result tuple to document."""
        doc_id, score, *rest = result
        hamming = rest[0] if len(rest) > 1 else 64
        payload = rest[-1] if rest else {}
        
        return RetrievedDocumentV2(
            id=doc_id,
            score=score,
            type=payload.get("type", "unknown"),
            url=payload.get("url"),
            caption=payload.get("caption"),
            text=payload.get("text"),
            title=payload.get("title"),
            metadata={
                k: v for k, v in payload.items() 
                if k not in ["type", "url", "caption", "text", "title", "fingerprint"]
            },
            match_type=match_type,
            fingerprint_distance=hamming if match_type == "exact" else 64
        )
    
    def _to_document_with_info(
        self,
        doc_id: str,
        score: float,
        payload: dict,
        match_info: dict
    ) -> RetrievedDocumentV2:
        """Convert result with match info to document."""
        
        # Determine primary match type
        strategies = match_info.get("strategies", [])
        hamming = match_info.get("hamming_distance", 64)
        
        # Extract all cosine-based strategy scores
        visual_score = match_info.get("visual_score", 0.0)
        textual_score = match_info.get("textual_score", 0.0)
        combined_text_score = match_info.get("combined_text_score", 0.0)
        combined_score = match_info.get("combined_score", 0.0)
        
        # Determine match type based on strongest cosine score
        if hamming < 5:
            match_type = "exact"
        elif visual_score > 0.7 and visual_score >= max(textual_score, combined_text_score, combined_score):
            match_type = "visual"
        elif textual_score > 0.7 and textual_score >= max(visual_score, combined_text_score, combined_score):
            match_type = "textual"
        elif combined_text_score > 0.55 or combined_score > 0.55:
            match_type = "semantic"
        elif max(visual_score, textual_score, combined_text_score, combined_score) > 0:
            match_type = "semantic"
        else:
            match_type = "semantic"
        
        # Include ALL strategy scores in metadata for confidence calculation
        extra_metadata = {
            k: v for k, v in payload.items()
            if k not in ["type", "url", "caption", "text", "title", "fingerprint"]
        }
        
        # Preserve all cosine-based scores for downstream confidence calculation
        extra_metadata["visual_score"] = visual_score
        extra_metadata["textual_score"] = textual_score
        extra_metadata["combined_text_score"] = combined_text_score
        extra_metadata["combined_score"] = combined_score
        extra_metadata["strategies"] = strategies
        
        return RetrievedDocumentV2(
            id=doc_id,
            score=score,
            type=payload.get("type", "unknown"),
            url=payload.get("url"),
            caption=payload.get("caption"),
            text=payload.get("text"),
            title=payload.get("title"),
            metadata=extra_metadata,
            match_type=match_type,
            visual_score=visual_score,
            textual_score=textual_score,
            fingerprint_distance=hamming
        )


# Singleton
enhanced_retriever = HybridRetrieverV2()