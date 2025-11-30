"""
Enhanced Ingestion Pipeline
Stores multiple vectors per document for robust retrieval.

Key Improvements:
1. Multi-vector storage (image_only, text_only, combined)
2. Perceptual hash storage for exact matching
3. Structured metadata extraction
4. Better OCR handling
5. Robust error handling and memory management for PDFs
"""
import uuid
import asyncio
import base64
import gc  # Added for PDF memory management
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams, 
    SparseIndexParams, PointStruct, SparseVector,
    PayloadSchemaType
)

from config import config
from storage import storage
from embeddings import (
    voyage_embedder_v2, 
    sparse_embedder_v2,
    MultiVectorEmbedding,
    ImageFingerprint
)
from llm_client import qwen_client


class DocumentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"


@dataclass
class IngestedDocumentV2:
    """Enhanced document with multi-vector info."""
    id: str
    type: DocumentType
    url: Optional[str] = None
    caption: Optional[str] = None
    text: Optional[str] = None
    fingerprint: Optional[Dict] = None  # Perceptual hash data
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class MultiVectorStore:
    """
    Qdrant store with multiple named vectors per document.
    
    Vector Configuration:
    - "image_dense": Pure visual embedding (1024-dim)
    - "text_dense": Pure text/caption embedding (1024-dim)
    - "combined_dense": Multimodal embedding (1024-dim)
    - "sparse": BM25 sparse vector
    """
    
    VECTOR_NAMES = {
        "image": "image_dense",
        "text": "text_dense", 
        "combined": "combined_dense",
        "sparse": "sparse"
    }
    
    def __init__(self):
        self._client = None
        self._collection_ready = False
    
    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                url=config.qdrant.url,
                api_key=config.qdrant.api_key or None
            )
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None
    
    async def ensure_collection(self):
        """Create collection with multi-vector configuration."""
        if self._collection_ready:
            return
        
        collection_name = f"{config.qdrant.collection_name}_v2"
        
        try:
            # Check if collection exists
            exists = await self.client.collection_exists(collection_name)
            
            if not exists:
                print(f"Creating collection {collection_name} with multi-vector config...")
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        # Image-only vector for visual search
                        self.VECTOR_NAMES["image"]: VectorParams(
                            size=config.embedding.dimension,
                            distance=Distance.COSINE
                        ),
                        # Text-only vector for caption/OCR search
                        self.VECTOR_NAMES["text"]: VectorParams(
                            size=config.embedding.dimension,
                            distance=Distance.COSINE
                        ),
                        # Combined multimodal vector
                        self.VECTOR_NAMES["combined"]: VectorParams(
                            size=config.embedding.dimension,
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        self.VECTOR_NAMES["sparse"]: SparseVectorParams(
                            index=SparseIndexParams(on_disk=False)
                        )
                    }
                )
                
                # Create payload indices for fast filtering
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="type",
                    field_schema=PayloadSchemaType.KEYWORD
                )
                await self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name="fingerprint.phash",
                    field_schema=PayloadSchemaType.KEYWORD
                )
            
            self._collection_ready = True
        except Exception as e:
            print(f"Ã¢ÂÅ’ Failed to ensure collection: {e}")
            raise
    
    @property
    def collection_name(self) -> str:
        return f"{config.qdrant.collection_name}_v2"
    
    async def upsert_multivector(
        self,
        doc_id: str,
        mv_embedding: MultiVectorEmbedding,
        sparse_vector: dict,
        payload: dict
    ):
        """
        Upsert document with multiple vectors.
        
        Handles cases where not all vectors are available:
        - Text documents: only text_dense and sparse
        - Image documents: all vectors
        """
        await self.ensure_collection()
        
        vectors = {}
        
        # Add available dense vectors
        if mv_embedding.image_only is not None:
            vectors[self.VECTOR_NAMES["image"]] = mv_embedding.image_only.tolist()
        
        if mv_embedding.text_only is not None:
            vectors[self.VECTOR_NAMES["text"]] = mv_embedding.text_only.tolist()
        
        if mv_embedding.combined is not None:
            vectors[self.VECTOR_NAMES["combined"]] = mv_embedding.combined.tolist()
        elif mv_embedding.image_only is not None:
            # Fallback: use image_only for combined if no text
            vectors[self.VECTOR_NAMES["combined"]] = mv_embedding.image_only.tolist()
        elif mv_embedding.text_only is not None:
            # Fallback: use text_only for combined if no image
            vectors[self.VECTOR_NAMES["combined"]] = mv_embedding.text_only.tolist()
        
        # Add sparse vector
        if sparse_vector and sparse_vector.get("indices"):
            vectors[self.VECTOR_NAMES["sparse"]] = SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"]
            )
        
        # Add fingerprint to payload
        if mv_embedding.fingerprint:
            payload["fingerprint"] = mv_embedding.fingerprint.to_dict()
        
        point = PointStruct(
            id=doc_id,
            vector=vectors,
            payload=payload
        )
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )


class EnhancedIngestionPipeline:
    """
    Improved ingestion with multi-vector storage.
    """
    
    def __init__(self):
        self.vector_store = MultiVectorStore()
    
    async def close(self):
        await self.vector_store.close()
        await voyage_embedder_v2.close()
    
    async def ingest_image(
        self,
        image_data: bytes,
        filename: str = None,
        metadata: dict = None,
        extracted_text: str = None,
        skip_caption: bool = False
    ) -> IngestedDocumentV2:
        """
        Ingest image with multi-vector embeddings.
        
        Process:
        1. Store image in object storage
        2. Generate caption via VLM (unless skipped)
        3. Create multi-vector embeddings (image-only, text-only, combined)
        4. Compute perceptual hash
        5. Store in Qdrant with all vectors
        """
        doc_id = str(uuid.uuid4())
        metadata = metadata or {}
        
        print(f"[Ingestion] Starting ingest for: {filename or 'unnamed'} ({len(image_data):,} bytes)")
        
        # 1. Store image
        stored = await storage.upload(image_data, filename)
        print(f"[Ingestion] Stored at: {stored.url}")
        
        # 2. Generate caption (with structured extraction for visual grounding)
        caption_for_embedding = None
        caption_display = None
        structured_data = {}
        
        if not skip_caption:
            print(f"[Ingestion] Generating structured caption...")
            # Retry logic for VLM calls
            for attempt in range(2):
                try:
                    # Get page number from metadata if available (for PDFs)
                    page_number = metadata.get("page_number") if metadata else None
                    
                    caption_for_embedding, structured_data = await self._generate_structured_caption(
                        image_data, 
                        filename,
                        stored.url,
                        page_number=page_number
                    )
                    
                    # Use description for display, full caption for embedding
                    caption_display = structured_data.get("description", caption_for_embedding)
                    
                    print(f"[Ingestion] Caption: {caption_display[:150]}..." if len(caption_display) > 150 else f"[Ingestion] Caption: {caption_display}")
                    break
                except Exception as e:
                    print(f"Warning: Caption attempt {attempt+1} failed: {e}")
                    if attempt == 1:
                        print("Skipping caption generation due to repeated errors.")
                        caption_for_embedding = "Image description unavailable."
                        caption_display = caption_for_embedding
                        structured_data = {}
        
        # 3. Generate multi-vector embeddings
        # The caption_for_embedding includes contextual prefix for better retrieval
        try:
            mv_embedding = await voyage_embedder_v2.encode_document_multivector(
                image=image_data,
                caption=caption_for_embedding,
                extracted_text=extracted_text
            )
        except Exception as e:
            print(f"ERROR: Embedding generation failed: {e}")
            raise
        
        # 4. Generate sparse vector from text content
        text_for_sparse = f"{caption_for_embedding or ''}\n{extracted_text or ''}".strip()
        sparse_vector = sparse_embedder_v2.encode(text_for_sparse)
        
        # 5. Build payload with structured data for visual grounding
        payload = {
            "type": DocumentType.IMAGE.value,
            "url": stored.url,
            "caption": caption_display,  # Display caption (without prefix)
            "caption_full": caption_for_embedding,  # Full caption with contextual prefix
            "extracted_text": extracted_text,
            "filename": filename,
            "storage_id": stored.id,
            "created_at": datetime.utcnow().isoformat(),
            # Structured data for visual grounding
            "components": structured_data.get("components", []),
            "document_type": structured_data.get("document_type", ""),
            "key_topics": structured_data.get("key_topics", []),
            **metadata
        }
        
        # 6. Store with multi-vectors
        await self.vector_store.upsert_multivector(
            doc_id=doc_id,
            mv_embedding=mv_embedding,
            sparse_vector=sparse_vector,
            payload=payload
        )
        
        # Log vectors stored
        vectors_stored = []
        if mv_embedding.image_only is not None:
            vectors_stored.append("image_dense")
        if mv_embedding.text_only is not None:
            vectors_stored.append("text_dense")
        if mv_embedding.combined is not None:
            vectors_stored.append("combined_dense")
        if sparse_vector.get("indices"):
            vectors_stored.append(f"sparse({len(sparse_vector['indices'])} terms)")
        print(f"[Ingestion] Stored vectors: {vectors_stored}")
        
        # Log component extraction
        num_components = len(structured_data.get("components", []))
        if num_components > 0:
            print(f"[Ingestion] Visual grounding: {num_components} components indexed")
        
        print(f"[Ingestion] Complete! doc_id={doc_id}")
        
        return IngestedDocumentV2(
            id=doc_id,
            type=DocumentType.IMAGE,
            url=stored.url,
            caption=caption_display,
            text=extracted_text,
            fingerprint=mv_embedding.fingerprint.to_dict() if mv_embedding.fingerprint else None,
            metadata={**(metadata or {}), "components": structured_data.get("components", [])}
        )
    
    async def _generate_structured_caption(
        self,
        image_data: bytes,
        filename: str,
        stored_url: str,
        page_number: int = None
    ) -> tuple:
        """
        Generate comprehensive caption with structured extraction.
        
        Returns:
            tuple: (caption_for_embedding, structured_data)
            
        The caption_for_embedding includes the contextual retrieval prefix
        which improves retrieval accuracy by 67% (Anthropic research).
        
        The structured_data includes:
        - components: List of identified components with bounding boxes
        - document_type: Classification of the page type
        - key_topics: Main topics covered
        """
        # Handle localhost URLs (Cloud VLM can't reach localhost)
        caption_input = stored_url
        if "localhost" in stored_url or "127.0.0.1" in stored_url:
            mime_type = "image/jpeg"
            if filename:
                ext = filename.lower().split('.')[-1]
                mime_map = {'png': 'image/png', 'webp': 'image/webp', 'gif': 'image/gif'}
                mime_type = mime_map.get(ext, 'image/jpeg')
            
            b64_data = base64.b64encode(image_data).decode('utf-8')
            caption_input = f"data:{mime_type};base64,{b64_data}"
        
        # Try structured extraction first (enables visual grounding + contextual retrieval)
        try:
            structured = await qwen_client.caption_image_structured(
                image_url=caption_input,
                filename=filename,
                page_number=page_number
            )
            
            # Build caption with contextual prefix for better retrieval
            context_prefix = structured.get("context_prefix", "")
            description = structured.get("description", "")
            transcribed_text = structured.get("transcribed_text", "")
            
            # Combine for embedding: prefix + description + transcribed text
            caption_for_embedding = f"{context_prefix}{description}"
            if transcribed_text:
                caption_for_embedding += f"\n\n[Text on page]: {transcribed_text}"
            
            print(f"[Ingestion] Structured extraction: {len(structured.get('components', []))} components detected")
            
            return caption_for_embedding, structured
            
        except Exception as e:
            print(f"⚠️ Structured caption failed, falling back to simple: {e}")
            
            # Fallback to simple captioning
            caption = await qwen_client.caption_image(
                image_url=caption_input,
                detail_level="high"
            )
            
            # Create minimal structured data
            simple_prefix = f"[Source: {filename or 'Document'}"
            if page_number:
                simple_prefix += f", Page {page_number}"
            simple_prefix += "] "
            
            return f"{simple_prefix}{caption}", {
                "description": caption,
                "transcribed_text": "",
                "context_prefix": simple_prefix,
                "components": []
            }
    
    async def ingest_text(
        self,
        text: str,
        title: str = None,
        metadata: dict = None
    ) -> IngestedDocumentV2:
        """Ingest text document."""
        doc_id = str(uuid.uuid4())
        metadata = metadata or {}
        
        # Create text-only embedding
        mv_embedding = MultiVectorEmbedding(
            text_only=await voyage_embedder_v2.encode_text_only(text, input_type="document")
        )
        
        # Sparse vector
        sparse_vector = sparse_embedder_v2.encode(text)
        
        payload = {
            "type": DocumentType.TEXT.value,
            "text": text,
            "title": title,
            "created_at": datetime.utcnow().isoformat(),
            **metadata
        }
        
        await self.vector_store.upsert_multivector(
            doc_id=doc_id,
            mv_embedding=mv_embedding,
            sparse_vector=sparse_vector,
            payload=payload
        )
        
        return IngestedDocumentV2(
            id=doc_id,
            type=DocumentType.TEXT,
            text=text,
            metadata=metadata
        )
    
    async def ingest_file(self, file_path: str, metadata: dict = None) -> List[IngestedDocumentV2]:
        """Ingest file with streaming progress support."""
        docs = []
        async for update in self.ingest_file_stream(file_path, metadata):
            if update["type"] == "complete":
                for d in update["docs"]:
                    docs.append(IngestedDocumentV2(
                        id=d["id"],
                        type=DocumentType(d["type"]),
                        url=d.get("url"),
                        caption=d.get("caption"),
                        fingerprint=d.get("fingerprint"),
                        metadata={"filename": d.get("filename")}
                    ))
        return docs
    
    async def ingest_file_stream(self, file_path: str, metadata: dict = None):
        """Async generator yielding progress updates."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            try:
                import fitz  # PyMuPDF
            except ImportError:
                yield {"type": "error", "message": "PyMuPDF not installed. Cannot process PDF."}
                return

            docs = []
            try:
                pdf = fitz.open(path)
                total_pages = len(pdf)
                
                yield {"type": "progress", "value": 0, "message": f"Opened PDF: {total_pages} pages"}
                
                for page_num, page in enumerate(pdf):
                    percent = int((page_num / total_pages) * 100)
                    yield {
                        "type": "progress", 
                        "value": percent, 
                        "message": f"Processing page {page_num + 1}/{total_pages}..."
                    }
                    
                    page_metadata = {
                        **(metadata or {}),
                        "source_file": path.name,
                        "page_number": page_num + 1,
                        "total_pages": total_pages
                    }
                    
                    # Extract text
                    raw_text = page.get_text()
                    
                    # Render page as image (High DPI for better OCR)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    image_data = pix.tobytes("png")
                    
                    # Explicit cleanup to prevent memory leaks in large PDFs
                    pix = None
                    
                    try:
                        doc = await self.ingest_image(
                            image_data=image_data,
                            filename=f"{path.stem}_p{page_num + 1}.png",
                            metadata=page_metadata,
                            extracted_text=raw_text.strip() if raw_text.strip() else None
                        )
                        docs.append(doc)
                        
                        # Rate limiting to prevent API throttling
                        await asyncio.sleep(1.0)
                        
                        # Force garbage collection periodically
                        if page_num % 5 == 0:
                            gc.collect()
                        
                    except Exception as e:
                        yield {"type": "log", "message": f"Warning: Failed page {page_num+1}: {e}"}
                
                pdf.close()
                
                doc_dicts = [{
                    "id": d.id,
                    "type": d.type.value,
                    "url": d.url,
                    "caption": d.caption,
                    "fingerprint": d.fingerprint,
                    "filename": d.metadata.get("filename")
                } for d in docs]
                
                yield {"type": "complete", "docs": doc_dicts}
                
            except Exception as e:
                yield {"type": "error", "message": f"PDF processing failed: {str(e)}"}
        
        elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            try:
                yield {"type": "progress", "value": 50, "message": "Processing image..."}
                
                with open(path, 'rb') as f:
                    image_data = f.read()
                
                doc = await self.ingest_image(image_data, path.name, metadata)
                
                yield {
                    "type": "complete",
                    "docs": [{
                        "id": doc.id,
                        "type": doc.type.value,
                        "url": doc.url,
                        "caption": doc.caption,
                        "fingerprint": doc.fingerprint,
                        "filename": path.name
                    }]
                }
                
            except Exception as e:
                yield {"type": "error", "message": f"Image ingestion failed: {str(e)}"}
        
        elif suffix == '.txt':
            try:
                yield {"type": "progress", "value": 50, "message": "Processing text..."}
                
                text = path.read_text(encoding='utf-8', errors='ignore')
                doc = await self.ingest_text(text, path.stem, metadata)
                
                yield {
                    "type": "complete",
                    "docs": [{
                        "id": doc.id,
                        "type": doc.type.value,
                        "caption": None,
                        "filename": path.name
                    }]
                }
                
            except Exception as e:
                yield {"type": "error", "message": f"Text ingestion failed: {str(e)}"}
        
        else:
            yield {"type": "error", "message": f"Unsupported file type: {suffix}"}


# Singleton
ingestion_pipeline = EnhancedIngestionPipeline()