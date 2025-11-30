
"""
Ingestion Pipeline
Dual-path indexing: Multimodal embedding (Voyage) + Caption (BM25 fallback)
"""
import uuid
import asyncio
import base64
import sys
from pathlib import Path
from typing import Union, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams, PointStruct, SparseVector

from config import config
from storage import storage, StoredImage
from embeddings import voyage_embedder, sparse_embedder
from llm_client import qwen_client


class DocumentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"


@dataclass
class IngestedDocument:
    id: str
    type: DocumentType
    url: Optional[str] = None
    caption: Optional[str] = None
    text: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


class VectorStore:
    """
    Async Qdrant vector store.
    """
    
    def __init__(self):
        self._client = None
        self._collection_exists = False
    
    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(url=config.qdrant.url)
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None
    
    async def ensure_collection(self):
        if self._collection_exists:
            return
        
        response = await self.client.get_collections()
        exists = any(c.name == config.qdrant.collection_name for c in response.collections)
        
        if not exists:
            await self.client.create_collection(
                collection_name=config.qdrant.collection_name,
                vectors_config={
                    config.qdrant.dense_vector_name: VectorParams(
                        size=config.embedding.dimension,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    config.qdrant.sparse_vector_name: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }
            )
        
        self._collection_exists = True
    
    async def upsert(self, doc_id: str, dense_vector: list, sparse_vector: dict, payload: dict):
        await self.ensure_collection()
        
        point = PointStruct(
            id=doc_id,
            vector={
                config.qdrant.dense_vector_name: dense_vector
            },
            payload=payload
        )
        
        if sparse_vector and sparse_vector.get("indices"):
            point.vector[config.qdrant.sparse_vector_name] = SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"]
            )
        
        await self.client.upsert(
            collection_name=config.qdrant.collection_name,
            points=[point]
        )


class IngestionPipeline:
    
    def __init__(self):
        self.vector_store = VectorStore()
    
    async def close(self):
        await self.vector_store.close()
    
    async def ingest_image(
        self,
        image_data: bytes,
        filename: str = None,
        metadata: dict = None,
        extracted_text: str = None
    ) -> IngestedDocument:
        doc_id = str(uuid.uuid4())
        metadata = metadata or {}
        
        stored = await storage.upload(image_data, filename)
        
        # Handle Localhost for Qwen
        caption_input = stored.url
        if "localhost" in stored.url or "127.0.0.1" in stored.url:
            mime_type = "image/jpeg"
            if filename:
                ext = filename.lower().split('.')[-1]
                if ext == 'png': mime_type = "image/png"
                elif ext == 'webp': mime_type = "image/webp"
                elif ext == 'gif': mime_type = "image/gif"
            
            b64_data = base64.b64encode(image_data).decode('utf-8')
            caption_input = f"data:{mime_type};base64,{b64_data}"

        caption = await qwen_client.caption_image(
            image_url=caption_input,
            detail_level="high"
        )
        
        dense_vector = await voyage_embedder.encode_multimodal(
            text=caption,
            image=image_data,
            input_type="document"
        )
        dense_vector = dense_vector.tolist()
        
        combined_text = f"{extracted_text or ''}\n\n{caption}".strip()
        sparse_vector = sparse_embedder.encode(combined_text)
        
        payload = {
            "type": DocumentType.IMAGE.value,
            "url": stored.url,
            "caption": caption,
            "text": extracted_text,
            "filename": filename,
            "storage_id": stored.id,
            "created_at": datetime.utcnow().isoformat(),
            **metadata
        }
        
        await self.vector_store.upsert(
            doc_id=doc_id,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            payload=payload
        )
        
        return IngestedDocument(
            id=doc_id,
            type=DocumentType.IMAGE,
            url=stored.url,
            caption=caption,
            text=extracted_text,
            metadata=metadata
        )
    
    async def ingest_text(
        self,
        text: str,
        title: str = None,
        metadata: dict = None
    ) -> IngestedDocument:
        doc_id = str(uuid.uuid4())
        metadata = metadata or {}
        
        dense_vector = await voyage_embedder.encode_text(
            text,
            input_type="document"
        )
        dense_vector = dense_vector.tolist()
        
        sparse_vector = sparse_embedder.encode(text)
        
        payload = {
            "type": DocumentType.TEXT.value,
            "text": text,
            "title": title,
            "created_at": datetime.utcnow().isoformat(),
            **metadata
        }
        
        await self.vector_store.upsert(
            doc_id=doc_id,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            payload=payload
        )
        
        return IngestedDocument(
            id=doc_id,
            type=DocumentType.TEXT,
            text=text,
            metadata=metadata
        )
    
    async def ingest_file(self, file_path: str, metadata: dict = None) -> List[IngestedDocument]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
            with open(path, 'rb') as f:
                image_data = f.read()
            doc = await self.ingest_image(image_data, path.name, metadata)
            return [doc]
        
        elif suffix == '.txt':
            text = path.read_text()
            doc = await self.ingest_text(text, path.stem, metadata)
            return [doc]
        
        elif suffix == '.pdf':
            docs = []
            async for update in self.ingest_file_stream(file_path, metadata):
                if update["type"] == "complete":
                    for d in update["docs"]:
                        docs.append(IngestedDocument(
                            id=d["id"],
                            type=DocumentType(d["type"]),
                            url=d["url"],
                            caption=d["caption"],
                            metadata={"filename": d["filename"]}
                        ))
            return docs
        
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    async def ingest_file_stream(self, file_path: str, metadata: dict = None):
        """
        Async generator that yields progress updates.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            import fitz
            docs = []
            try:
                pdf = fitz.open(path)
                total_pages = len(pdf)
                
                yield {"type": "progress", "value": 0, "message": f"Opened PDF: {total_pages} pages."}
                
                for page_num, page in enumerate(pdf):
                    percent = int(((page_num) / total_pages) * 100)
                    yield {"type": "progress", "value": percent, "message": f"Processing page {page_num + 1}/{total_pages}..."}
                    
                    page_metadata = {**(metadata or {}), "source_file": path.name, "page_number": page_num + 1}
                    
                    raw_text = page.get_text()
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    image_data = pix.tobytes("png")
                    
                    try:
                        doc = await self.ingest_image(
                            image_data=image_data,
                            filename=f"{path.stem}_p{page_num + 1}.png",
                            metadata=page_metadata,
                            extracted_text=raw_text.strip() if raw_text.strip() else None
                        )
                        docs.append(doc)
                        await asyncio.sleep(2.0)
                    except Exception as e:
                        yield {"type": "log", "message": f"Warning: Failed page {page_num+1}: {e}"}
                
                pdf.close()
                
                doc_dicts = [{
                    "id": d.id, "type": d.type.value, "url": d.url, 
                    "caption": d.caption, "filename": d.metadata.get("filename")
                } for d in docs]
                
                yield {"type": "complete", "docs": doc_dicts}

            except Exception as e:
                yield {"type": "error", "message": f"PDF Processing failed: {str(e)}"}

        else:
            try:
                yield {"type": "progress", "value": 50, "message": "Processing file..."}
                
                if suffix in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
                    with open(path, 'rb') as f:
                        image_data = f.read()
                    doc = await self.ingest_image(image_data, path.name, metadata)
                    result_docs = [doc]
                elif suffix == '.txt':
                    text = path.read_text()
                    doc = await self.ingest_text(text, path.stem, metadata)
                    result_docs = [doc]
                else:
                    raise ValueError(f"Unsupported file type: {suffix}")
                
                doc_dicts = [{
                    "id": d.id, "type": d.type.value, "url": d.url, 
                    "caption": d.caption, "filename": d.metadata.get("filename")
                } for d in result_docs]
                
                yield {"type": "complete", "docs": doc_dicts}
            except Exception as e:
                yield {"type": "error", "message": f"Ingestion failed: {str(e)}"}


# Singleton instance
ingestion_pipeline = IngestionPipeline()