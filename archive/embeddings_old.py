"""
Multimodal Embedding Module
Uses Voyage Multimodal-3 for unified text-image embedding space.
"""
import base64
import asyncio
import mimetypes
from io import BytesIO
from typing import Union, List, Tuple
from pathlib import Path
from collections import Counter
import re
import time

import numpy as np
from PIL import Image
import httpx

from config import config
from usage import tracker

class VoyageMultimodalEmbedder:
    """
    Unified text-image embeddings using Voyage Multimodal-3 API.
    """
    
    def __init__(self):
        self.api_key = config.voyage.api_key
        self.model = config.voyage.model_name
        self.api_url = "https://api.voyageai.com/v1/multimodalembeddings"
        self._client = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=60.0,  # Increased timeout
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._client
    
    def _process_image(self, image_input: Union[str, bytes, Image.Image]) -> Tuple[str, int]:
        """
        Convert image to base64 and calculate pixels.
        Returns: (base64_string, pixel_count)
        """
        # 1. Load Image
        if isinstance(image_input, Image.Image):
            img = image_input
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
        
        elif isinstance(image_input, bytes):
            img = Image.open(BytesIO(image_input))
        
        elif isinstance(image_input, str):
            if image_input.startswith("data:"):
                try:
                    header, encoded = image_input.split(",", 1)
                    data = base64.b64decode(encoded)
                    img = Image.open(BytesIO(data))
                except Exception:
                    data = base64.b64decode(image_input)
                    img = Image.open(BytesIO(data))
            else:
                path = Path(image_input)
                if path.exists():
                    img = Image.open(path)
                else:
                    try:
                        data = base64.b64decode(image_input)
                        img = Image.open(BytesIO(data))
                    except:
                        raise ValueError("Could not process image input")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # 2. Resize if necessary (Voyage limit)
        total_pixels = img.width * img.height
        if total_pixels > config.voyage.max_image_pixels:
            scale = (config.voyage.max_image_pixels / total_pixels) ** 0.5
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            total_pixels = img.width * img.height

        # 3. Convert to Base64
        buffer = BytesIO()
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        img.save(buffer, format='JPEG', quality=85)
        image_bytes = buffer.getvalue()
        
        encoded = base64.b64encode(image_bytes).decode('utf-8')
        final_b64 = f"data:image/jpeg;base64,{encoded}"
        
        return final_b64, total_pixels
    
    async def _call_api(
        self,
        inputs: List[dict],
        input_type: str = None,
        pixel_count: int = 0
    ) -> List[List[float]]:
        """Call Voyage API and track usage with robust rate limiting."""
        payload = {
            "model": self.model,
            "inputs": inputs,
            "truncation": True
        }
        
        if input_type:
            payload["input_type"] = input_type
        
        # Increase retries for batch processing
        max_retries = 8 
        
        for attempt in range(max_retries):
            try:
                response = await self.client.post(self.api_url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # --- TRACK USAGE ---
                tokens = 0
                if "usage" in data and "total_tokens" in data["usage"]:
                    tokens = data["usage"]["total_tokens"]
                
                tracker.track_voyage(tokens=tokens, pixels=pixel_count)
                # -------------------

                return data["data"]
            
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Handle Rate Limit
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = float(retry_after) + 1
                    else:
                        # Aggressive backoff: 5s, 10s, 20s...
                        wait_time = 5 * (2 ** attempt)
                    
                    print(f"⚠️ Rate limit hit (429). Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Voyage API Error: {e.response.text}")
                    raise
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                print(f"API Error: {str(e)}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        raise RuntimeError("Voyage API call failed after retries")
    
    async def encode_text(
        self,
        text: Union[str, List[str]],
        input_type: str = None
    ) -> np.ndarray:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        inputs = [{"content": [{"type": "text", "text": t}]} for t in texts]
        embeddings_data = await self._call_api(inputs, input_type, pixel_count=0)
        embeddings = [item["embedding"] for item in embeddings_data]
        return np.array(embeddings[0] if len(texts) == 1 else embeddings)
    
    async def encode_image(
        self,
        image: Union[str, bytes, Image.Image],
        input_type: str = None
    ) -> np.ndarray:
        base64_image, pixels = self._process_image(image)
        inputs = [{
            "content": [{"type": "image_base64", "image_base64": base64_image}]
        }]
        embeddings_data = await self._call_api(inputs, input_type, pixel_count=pixels)
        return np.array(embeddings_data[0]["embedding"])
    
    async def encode_multimodal(
        self,
        text: str = None,
        image: Union[str, bytes, Image.Image] = None,
        input_type: str = None
    ) -> np.ndarray:
        content = []
        total_pixels = 0
        
        if text:
            content.append({"type": "text", "text": text})
        
        if image:
            base64_image, pixels = self._process_image(image)
            total_pixels = pixels
            content.append({"type": "image_base64", "image_base64": base64_image})
        
        if not content:
            raise ValueError("Must provide either text or image")
        
        inputs = [{"content": content}]
        embeddings_data = await self._call_api(inputs, input_type, pixel_count=total_pixels)
        return np.array(embeddings_data[0]["embedding"])
    
    async def encode_query(self, text: str = None, image: Union[str, bytes, Image.Image] = None) -> np.ndarray:
        if text and image:
            return await self.encode_multimodal(text, image, input_type="query")
        elif image:
            return await self.encode_image(image, input_type="query")
        elif text:
            return await self.encode_text(text, input_type="query")
        else:
            raise ValueError("Must provide either text or image")
    
    async def encode_document(self, text: str = None, image: Union[str, bytes, Image.Image] = None) -> np.ndarray:
        if text and image:
            return await self.encode_multimodal(text, image, input_type="document")
        elif image:
            return await self.encode_image(image, input_type="document")
        elif text:
            return await self.encode_text(text, input_type="document")
        else:
            raise ValueError("Must provide either text or image")
    
    @property
    def dimension(self) -> int:
        return config.embedding.dimension
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

# Synchronous wrapper
class MultimodalEmbedder:
    def __init__(self):
        self._async_embedder = VoyageMultimodalEmbedder()
    
    def _run_async(self, coro):
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        return self._run_async(self._async_embedder.encode_text(text))
    
    def encode_image(self, image: Union[str, bytes, Image.Image]) -> np.ndarray:
        return self._run_async(self._async_embedder.encode_document(image=image))
    
    def encode_query(self, text: str = None, image: Union[str, bytes, Image.Image] = None) -> np.ndarray:
        return self._run_async(self._async_embedder.encode_query(text, image))
    
    @property
    def dimension(self) -> int:
        return config.embedding.dimension

class SparseEmbedder:
    def __init__(self):
        self._stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'it', 'its'
        }
    
    def encode(self, text: str) -> dict:
        import zlib
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        tokens = [t for t in tokens if t not in self._stopwords and len(t) > 2]
        token_counts = Counter(tokens)
        
        if not token_counts:
            return {"indices": [], "values": []}
        
        sparse_vector = {}
        for token, count in token_counts.items():
            token_bytes = token.encode('utf-8')
            idx = zlib.adler32(token_bytes) % 30000
            weight = 1.0 + np.log(count)
            if idx in sparse_vector:
                sparse_vector[idx] += weight
            else:
                sparse_vector[idx] = weight
        
        return {
            "indices": list(sparse_vector.keys()),
            "values": list(sparse_vector.values())
        }

voyage_embedder = VoyageMultimodalEmbedder()
embedder = MultimodalEmbedder()
sparse_embedder = SparseEmbedder()