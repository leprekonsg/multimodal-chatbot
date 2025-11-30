"""
Enhanced Multimodal Embedding Module
Addresses: Embedding asymmetry, perceptual hashing, multi-vector storage
Optimized for: Strict Rate Limiting (3 RPM, 10k TPM)

Key Improvements:
1. Batching: Combines Image/Text/Multimodal vectors into 1 API call per doc.
2. Rate Limiting: Client-side semaphore ensures <3 requests per minute.
3. Robustness: Retries on 429 errors with exponential backoff.
4. Consistency: Robust image preprocessing for all input types.
"""
import base64
import asyncio
import time
import hashlib
from io import BytesIO
from typing import Union, List, Tuple, Optional, Dict, Any
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
import re

import numpy as np
from PIL import Image
import httpx

from config import config
from usage import tracker


@dataclass
class ImageFingerprint:
    """Perceptual hash and metadata for image matching."""
    phash: str           # Perceptual hash (64-bit)
    dhash: str           # Difference hash (64-bit) 
    avg_color: Tuple[int, int, int]  # Average RGB
    dimensions: Tuple[int, int]       # Original dimensions
    
    def hamming_distance(self, other: 'ImageFingerprint') -> int:
        """
        Calculate combined hamming distance using multiple hashes.
        Uses minimum of pHash and dHash distances for robustness.
        """
        if len(self.phash) != len(other.phash):
            return 64  # Max distance
        
        # Calculate phash distance
        phash_dist = sum(c1 != c2 for c1, c2 in zip(self.phash, other.phash))
        
        # Calculate dhash distance
        dhash_dist = sum(c1 != c2 for c1, c2 in zip(self.dhash, other.dhash))
        
        # Also consider color difference as a tiebreaker
        color_diff = sum(abs(a - b) for a, b in zip(self.avg_color, other.avg_color))
        color_dist = min(10, color_diff // 30)  # Normalize to 0-10 range
        
        # Use combined distance (weighted average)
        combined = (phash_dist + dhash_dist) // 2 + color_dist // 2
        return combined
    
    def is_near_duplicate(self, other: 'ImageFingerprint', threshold: int = 12) -> bool:
        """Check if images are near-duplicates (hamming distance < threshold)."""
        return self.hamming_distance(other) < threshold
    
    def to_dict(self) -> dict:
        return {
            "phash": self.phash,
            "dhash": self.dhash,
            "avg_color": list(self.avg_color),
            "dimensions": list(self.dimensions)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ImageFingerprint':
        return cls(
            phash=data["phash"],
            dhash=data["dhash"],
            avg_color=tuple(data["avg_color"]),
            dimensions=tuple(data["dimensions"])
        )


@dataclass
class MultiVectorEmbedding:
    """
    Multi-vector representation for a document.
    Solves the asymmetry problem by storing multiple vectors.
    """
    # Primary vectors
    image_only: Optional[np.ndarray] = None      # Pure visual embedding
    text_only: Optional[np.ndarray] = None       # Pure text/caption embedding
    combined: Optional[np.ndarray] = None        # Fused multimodal embedding
    
    # Perceptual fingerprint for exact matching
    fingerprint: Optional[ImageFingerprint] = None
    
    # Metadata
    original_text: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None


class PerceptualHasher:
    """
    Generate perceptual hashes for images.
    Used for fast exact/near-duplicate detection.
    """
    
    @staticmethod
    def compute_phash(img: Image.Image, hash_size: int = 8) -> str:
        """Compute perceptual hash using average hash approach."""
        img_gray = img.convert('L').resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img_gray, dtype=np.float64)
        avg = pixels.mean()
        diff = pixels > avg
        return ''.join(['1' if b else '0' for b in diff.flatten()])
    
    @staticmethod
    def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
        """Compute difference hash."""
        img_gray = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img_gray, dtype=np.float64)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return ''.join(['1' if b else '0' for b in diff.flatten()])
    
    @staticmethod
    def compute_average_color(img: Image.Image) -> Tuple[int, int, int]:
        """Compute average RGB color."""
        img = img.convert('RGB').resize((50, 50), Image.Resampling.LANCZOS)
        pixels = np.array(img)
        avg = pixels.mean(axis=(0, 1))
        return tuple(int(c) for c in avg)
    
    @classmethod
    def compute_fingerprint(cls, img: Image.Image) -> ImageFingerprint:
        """Compute complete image fingerprint."""
        return ImageFingerprint(
            phash=cls.compute_phash(img),
            dhash=cls.compute_dhash(img),
            avg_color=cls.compute_average_color(img),
            dimensions=(img.width, img.height)
        )


class ImagePreprocessor:
    """
    Consistent image preprocessing for both ingestion and query.
    Ensures the same image produces the same embedding.
    """
    
    # Standard size for embedding (maintains aspect ratio)
    TARGET_MAX_DIMENSION = 1024
    JPEG_QUALITY = 90
    
    @classmethod
    def normalize(cls, image_input: Union[str, bytes, Image.Image]) -> Tuple[bytes, Image.Image]:
        """
        Normalize image to consistent format.
        Returns: (normalized_bytes, PIL_Image)
        """
        try:
            # Load image
            if isinstance(image_input, Image.Image):
                img = image_input
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
                        data = base64.b64decode(image_input)
                        img = Image.open(BytesIO(data))
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Verify image integrity
            img.verify()
            # Re-open after verify (verify closes the file pointer in some versions)
            if isinstance(image_input, bytes):
                img = Image.open(BytesIO(image_input))
            elif isinstance(image_input, str) and not image_input.startswith("data:"):
                 img = Image.open(image_input)
            
            # Convert to RGB (handles transparency, grayscale, etc.)
            if img.mode == 'RGBA':
                # Handle transparency by compositing on white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (maintain aspect ratio)
            max_dim = max(img.width, img.height)
            if max_dim > cls.TARGET_MAX_DIMENSION:
                scale = cls.TARGET_MAX_DIMENSION / max_dim
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Ensure Voyage pixel limit
            total_pixels = img.width * img.height
            if total_pixels > config.voyage.max_image_pixels:
                scale = (config.voyage.max_image_pixels / total_pixels) ** 0.5
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to consistent JPEG bytes
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=cls.JPEG_QUALITY)
            normalized_bytes = buffer.getvalue()
            
            return normalized_bytes, img
            
        except Exception as e:
            print(f"âŒ Image normalization failed: {e}")
            # Create a blank 1x1 image to allow pipeline to fail gracefully downstream
            blank = Image.new('RGB', (1, 1), color='black')
            buf = BytesIO()
            blank.save(buf, format='JPEG')
            return buf.getvalue(), blank


class RateLimiter:
    """
    Strict client-side rate limiter for Voyage API.
    Enforces 3 RPM (Requests Per Minute) = 1 request every 20 seconds.
    """
    def __init__(self, rpm: int = 3):
        self.interval = 60.0 / rpm  # e.g., 20.0 seconds
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            
            if elapsed < self.interval:
                wait_time = self.interval - elapsed + 0.1  # +0.1s buffer
                print(f"â³ Rate limit (3 RPM): Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()


class VoyageMultimodalEmbedderV2:
    """
    Enhanced Voyage embedder with multi-vector support.
    
    Key Changes:
    1. encode_document_multivector() - Batched API call for Image+Text+Combined
    2. Rate Limiting - Enforces 3 RPM
    3. Consistent preprocessing via ImagePreprocessor
    """
    
    def __init__(self):
        self.api_key = config.voyage.api_key
        self.model = config.voyage.model_name
        self.api_url = "https://api.voyageai.com/v1/multimodalembeddings"
        self._client = None
        self.preprocessor = ImagePreprocessor()
        self.hasher = PerceptualHasher()
        # Initialize rate limiter (3 RPM)
        self.rate_limiter = RateLimiter(rpm=3)
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=120.0, # Increased timeout for batch processing
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._client
    
    def _to_base64(self, image_bytes: bytes) -> str:
        """Convert bytes to data URI."""
        encoded = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"
    
    async def _call_api(
        self,
        inputs: List[dict],
        input_type: str = None,
        pixel_count: int = 0
    ) -> List[List[float]]:
        """
        Call Voyage API with Rate Limiting and Retry logic.
        """
        # 1. Wait for Rate Limit Slot
        await self.rate_limiter.acquire()
        
        payload = {
            "model": self.model,
            "inputs": inputs,
            "truncation": True
        }
        
        if input_type:
            payload["input_type"] = input_type
        
        max_retries = 5
        input_summary = f"{len(inputs)} inputs, type={input_type or 'none'}"
        print(f"[Voyage API] Calling with {input_summary}...")
        
        for attempt in range(max_retries):
            try:
                response = await self.client.post(self.api_url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Track usage
                tokens = data.get("usage", {}).get("total_tokens", 0)
                tracker.track_voyage(tokens=tokens, pixels=pixel_count)
                
                print(f"[Voyage API] Success: {len(data['data'])} embeddings returned")
                return data["data"]
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # If we hit 429 despite our rate limiter, back off exponentially
                    retry_after = e.response.headers.get("Retry-After")
                    wait_time = float(retry_after) + 1 if retry_after else 20 * (2 ** attempt)
                    print(f"âš ï¸ API 429 Hit. Backing off {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Voyage API Error: {e.response.text}")
                    raise
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"âš ï¸ Connection error: {e}. Retrying...")
                await asyncio.sleep(5 * (attempt + 1))
        
        raise RuntimeError("Voyage API call failed after retries")
    
    async def encode_image_only(
        self,
        image: Union[str, bytes, Image.Image],
        input_type: str = "document"
    ) -> np.ndarray:
        """Encode image without any text context."""
        normalized_bytes, img = self.preprocessor.normalize(image)
        base64_image = self._to_base64(normalized_bytes)
        
        inputs = [{
            "content": [{"type": "image_base64", "image_base64": base64_image}]
        }]
        
        pixel_count = img.width * img.height
        embeddings_data = await self._call_api(inputs, input_type, pixel_count)
        return np.array(embeddings_data[0]["embedding"])
    
    async def encode_text_only(
        self,
        text: str,
        input_type: str = "document"
    ) -> np.ndarray:
        """Encode text without image."""
        inputs = [{"content": [{"type": "text", "text": text}]}]
        embeddings_data = await self._call_api(inputs, input_type, pixel_count=0)
        return np.array(embeddings_data[0]["embedding"])
    
    async def encode_multimodal(
        self,
        text: str,
        image: Union[str, bytes, Image.Image],
        input_type: str = "document"
    ) -> np.ndarray:
        """Encode combined text + image."""
        normalized_bytes, img = self.preprocessor.normalize(image)
        base64_image = self._to_base64(normalized_bytes)
        
        content = [
            {"type": "text", "text": text},
            {"type": "image_base64", "image_base64": base64_image}
        ]
        
        inputs = [{"content": content}]
        pixel_count = img.width * img.height
        embeddings_data = await self._call_api(inputs, input_type, pixel_count)
        return np.array(embeddings_data[0]["embedding"])
    
    async def encode_document_multivector(
        self,
        image: Union[str, bytes, Image.Image],
        caption: str = None,
        extracted_text: str = None
    ) -> MultiVectorEmbedding:
        """
        Generate multiple embeddings in a SINGLE API call to save RPM.
        
        We batch:
        1. Image Only
        2. Text Only (if exists)
        3. Combined (if exists)
        """
        normalized_bytes, img = self.preprocessor.normalize(image)
        fingerprint = self.hasher.compute_fingerprint(img)
        base64_image = self._to_base64(normalized_bytes)
        
        # Combine text content
        text_content = ""
        if caption:
            text_content += caption
        if extracted_text:
            text_content += f"\n\n[Extracted Text]\n{extracted_text}"
        text_content = text_content.strip()
        
        # --- BATCHING STRATEGY ---
        # We construct a single list of inputs to send in one HTTP request
        api_inputs = []
        
        # 1. Image Only Input
        api_inputs.append({
            "content": [{"type": "image_base64", "image_base64": base64_image}]
        })
        
        # 2. Text Only Input (if applicable)
        if text_content:
            api_inputs.append({
                "content": [{"type": "text", "text": text_content}]
            })
            
        # 3. Combined Input (if applicable)
        if text_content:
            api_inputs.append({
                "content": [
                    {"type": "text", "text": text_content},
                    {"type": "image_base64", "image_base64": base64_image}
                ]
            })
            
        # Execute Single API Call
        pixel_count = img.width * img.height
        # We multiply pixel count by number of image occurrences in batch for tracking
        total_pixels = pixel_count * (2 if text_content else 1)
        
        embeddings_data = await self._call_api(
            inputs=api_inputs, 
            input_type="document", 
            pixel_count=total_pixels
        )
        
        # Unpack Results
        mv = MultiVectorEmbedding(
            image_only=np.array(embeddings_data[0]["embedding"]),
            fingerprint=fingerprint,
            original_text=text_content
        )
        
        if text_content:
            # If text existed, we sent 3 items. 
            # Index 0: Image, Index 1: Text, Index 2: Combined
            mv.text_only = np.array(embeddings_data[1]["embedding"])
            mv.combined = np.array(embeddings_data[2]["embedding"])
        
        return mv
    
    async def encode_query_adaptive(
        self,
        text: str = None,
        image: Union[str, bytes, Image.Image] = None,
        query_intent: str = "general"  # "visual", "textual", "general"
    ) -> Dict[str, np.ndarray]:
        """
        Generate query embeddings that match the stored vectors.
        Batched where possible to save RPM.
        """
        result = {}
        api_inputs = []
        input_map = [] # To map result index back to key
        
        # Prepare Image Inputs
        if image is not None:
            normalized_bytes, img = self.preprocessor.normalize(image)
            base64_image = self._to_base64(normalized_bytes)
            result["fingerprint"] = self.hasher.compute_fingerprint(img)
            
            # Image Query
            api_inputs.append({
                "content": [{"type": "image_base64", "image_base64": base64_image}]
            })
            input_map.append("image_query")
            
            # Combined Query (if text exists)
            if text:
                api_inputs.append({
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_base64", "image_base64": base64_image}
                    ]
                })
                input_map.append("combined_query")

        # Prepare Text Inputs
        if text:
            api_inputs.append({
                "content": [{"type": "text", "text": text}]
            })
            input_map.append("text_query")
            
        if not api_inputs:
            return {}
            
        # Execute Batch Call
        embeddings_data = await self._call_api(
            inputs=api_inputs,
            input_type="query",
            pixel_count=0 # Query pixels usually don't count against ingestion quota same way
        )
        
        # Map results back
        for i, key in enumerate(input_map):
            result[key] = np.array(embeddings_data[i]["embedding"])
            
        return result
    
    @property
    def dimension(self) -> int:
        return config.embedding.dimension
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


class SparseEmbedderV2:
    """
    Enhanced sparse embedder with better tokenization.
    """
    
    def __init__(self):
        self._stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'he',
            'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    def encode(self, text: str) -> dict:
        """Generate sparse vector for BM25-style matching."""
        import zlib
        
        if not text:
            return {"indices": [], "values": []}
        
        # Tokenize
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        tokens = [t for t in tokens if t not in self._stopwords and len(t) > 2]
        
        # Also extract important patterns
        # Error codes, product names, numbers
        error_codes = re.findall(r'[A-Z]{2,}\d+|\d+[A-Z]+|ERR\w+|ERROR\s*\d+', text, re.IGNORECASE)
        tokens.extend([c.lower() for c in error_codes])
        
        token_counts = Counter(tokens)
        
        if not token_counts:
            return {"indices": [], "values": []}
        
        sparse_vector = {}
        for token, count in token_counts.items():
            token_bytes = token.encode('utf-8')
            idx = zlib.adler32(token_bytes) % 30000
            weight = 1.0 + np.log(count)
            
            # Boost error codes and specific patterns
            if re.match(r'^[a-z]+\d+$|^\d+[a-z]+$', token):
                weight *= 1.5  # Boost error codes
            
            sparse_vector[idx] = sparse_vector.get(idx, 0) + weight
        
        return {
            "indices": list(sparse_vector.keys()),
            "values": list(sparse_vector.values())
        }
    
    def encode_from_image_caption(self, caption: str, extracted_text: str = None) -> dict:
        """
        Generate sparse vector from image metadata.
        This enables sparse search even for image-only queries by using
        the caption as a proxy.
        """
        combined = caption or ""
        if extracted_text:
            combined += f" {extracted_text}"
        return self.encode(combined)


# Singleton instances
voyage_embedder_v2 = VoyageMultimodalEmbedderV2()
sparse_embedder_v2 = SparseEmbedderV2()