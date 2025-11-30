"""
Object Storage Handler
Stores images as URLs (not base64) to reduce latency and token bloat.
"""
import os
import uuid
import hashlib
import aiofiles
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from config import config


@dataclass
class StoredImage:
    """Stored image reference."""
    id: str
    url: str
    local_path: Optional[str] = None


class LocalStorage:
    """Local file storage with HTTP serving."""
    
    def __init__(self):
        self.base_path = Path(config.storage.local_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.public_url_base = config.storage.public_url_base
    
    async def upload(self, image_data: bytes, filename: Optional[str] = None) -> StoredImage:
        """Store image locally and return URL."""
        # Generate unique ID based on content hash
        content_hash = hashlib.md5(image_data).hexdigest()[:12]
        ext = Path(filename).suffix if filename else ".jpg"
        image_id = f"{content_hash}_{uuid.uuid4().hex[:8]}{ext}"
        
        local_path = self.base_path / image_id
        
        async with aiofiles.open(local_path, 'wb') as f:
            await f.write(image_data)
        
        url = f"{self.public_url_base}/{image_id}"
        
        return StoredImage(
            id=image_id,
            url=url,
            local_path=str(local_path)
        )
    
    async def get_url(self, image_id: str) -> Optional[str]:
        """Get URL for stored image."""
        local_path = self.base_path / image_id
        if local_path.exists():
            return f"{self.public_url_base}/{image_id}"
        return None


class S3Storage:
    """AWS S3 / Compatible storage (MinIO, etc.)."""
    
    def __init__(self):
        import boto3
        self.client = boto3.client(
            's3',
            endpoint_url=config.storage.endpoint or None,
            aws_access_key_id=config.storage.access_key,
            aws_secret_access_key=config.storage.secret_key
        )
        self.bucket = config.storage.bucket
    
    async def upload(self, image_data: bytes, filename: Optional[str] = None) -> StoredImage:
        """Upload to S3 and return public URL."""
        import asyncio
        
        content_hash = hashlib.md5(image_data).hexdigest()[:12]
        ext = Path(filename).suffix if filename else ".jpg"
        image_id = f"images/{content_hash}_{uuid.uuid4().hex[:8]}{ext}"
        
        # S3 upload is sync, run in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.put_object(
                Bucket=self.bucket,
                Key=image_id,
                Body=image_data,
                ContentType=f"image/{ext.lstrip('.')}"
            )
        )
        
        # Generate presigned URL or public URL
        url = f"https://{self.bucket}.s3.amazonaws.com/{image_id}"
        
        return StoredImage(id=image_id, url=url)
    
    async def get_url(self, image_id: str, expires_in: int = 3600) -> Optional[str]:
        """Get presigned URL for private buckets."""
        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': image_id},
                ExpiresIn=expires_in
            )
            return url
        except Exception:
            return None


class OSSStorage:
    """Alibaba Cloud OSS storage."""
    
    def __init__(self):
        import oss2
        auth = oss2.Auth(config.storage.access_key, config.storage.secret_key)
        self.bucket = oss2.Bucket(auth, config.storage.endpoint, config.storage.bucket)
    
    async def upload(self, image_data: bytes, filename: Optional[str] = None) -> StoredImage:
        """Upload to OSS and return URL."""
        import asyncio
        
        content_hash = hashlib.md5(image_data).hexdigest()[:12]
        ext = Path(filename).suffix if filename else ".jpg"
        image_id = f"images/{content_hash}_{uuid.uuid4().hex[:8]}{ext}"
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.bucket.put_object(image_id, image_data)
        )
        
        url = f"https://{config.storage.bucket}.{config.storage.endpoint}/{image_id}"
        
        return StoredImage(id=image_id, url=url)
    
    async def get_url(self, image_id: str) -> Optional[str]:
        """Get public URL for OSS object."""
        return f"https://{config.storage.bucket}.{config.storage.endpoint}/{image_id}"


def get_storage():
    """Factory function to get storage backend."""
    provider = config.storage.provider.lower()
    
    if provider == "s3":
        return S3Storage()
    elif provider == "oss":
        return OSSStorage()
    else:
        return LocalStorage()


# Singleton instance
storage = get_storage()
