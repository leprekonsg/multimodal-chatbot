"""
Configuration Module
Centralizes settings for Voyage Multimodal-3, Qdrant, Qwen, and Handoffs.
"""
import os
from dataclasses import dataclass, field
from typing import List
from enum import Enum

# Helper to get env vars with defaults or raise error
def get_env(key: str, default: str = None, required: bool = False) -> str:
    val = os.environ.get(key, default)
    if required and not val:
        raise ValueError(f"Missing required environment variable: {key}")
    return val

@dataclass
class VoyageConfig:
    api_key: str = field(default_factory=lambda: get_env("VOYAGE_API_KEY", required=True))
    model_name: str = "voyage-multimodal-3"
    dimension: int = 1024
    max_image_pixels: int = 16_000_000

@dataclass
class QwenConfig:
    api_key: str = field(default_factory=lambda: get_env("DASHSCOPE_API_KEY", required=True))
    # Standard OpenAI-compatible endpoint for Alibaba Cloud
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    caption_max_tokens: int = 500  # Increased for Qwen3's better detail
    max_output_tokens: int = 2000
    enable_thinking_default: bool = True # Qwen3-VL-Plus supports thinking

@dataclass
class QdrantConfig:
    url: str = field(default_factory=lambda: get_env("QDRANT_URL", "http://localhost:6333"))
    api_key: str = field(default_factory=lambda: get_env("QDRANT_API_KEY", ""))
    collection_name: str = "knowledge_base"
    top_k: int = 10
    hnsw_ef: int = 128

@dataclass
class StorageConfig:
    provider: str = field(default_factory=lambda: get_env("STORAGE_PROVIDER", "local"))
    local_path: str = "static/images"
    public_url_base: str = "http://localhost:8000/images"
    bucket: str = os.environ.get("STORAGE_BUCKET", "")
    endpoint: str = os.environ.get("STORAGE_ENDPOINT", "")
    access_key: str = os.environ.get("STORAGE_ACCESS_KEY", "")
    secret_key: str = os.environ.get("STORAGE_SECRET_KEY", "")

@dataclass
class EscalationConfig:
    low_confidence_threshold: float = 0.50
    warn_confidence_threshold: float = 0.65
    negative_sentiment_threshold: float = -0.6
    max_failed_attempts: int = 2
    escalation_phrases: List[str] = field(default_factory=lambda: [
        "talk to human", "speak to agent", "real person", "escalate", "manager"
    ])
    uncertainty_phrases: List[str] = field(default_factory=lambda: [
        "i don't have information", "i cannot find", "i'm not sure", "context doesn't mention"
    ])

@dataclass
class HandoffConfig:
    webhook_url: str = os.environ.get("HANDOFF_WEBHOOK_URL", "")
    webhook_api_key: str = os.environ.get("HANDOFF_WEBHOOK_KEY", "")
    slack_bot_token: str = os.environ.get("SLACK_BOT_TOKEN", "")
    slack_channel: str = os.environ.get("SLACK_CHANNEL", "")
    app_base_url: str = os.environ.get("APP_BASE_URL", "http://localhost:8000")

@dataclass
class UXConfig:
    show_source_images: bool = True
    max_sources_displayed: int = 4
    thinking_mode_triggers: List[str] = field(default_factory=lambda: [
        "analyze", "compare", "reason", "why", "how come", "calculate"
    ])

@dataclass
class EmbeddingConfig:
    dimension: int = 1024

# UPDATED: Using Qwen3-VL models
class ModelTier(Enum):
    FLASH = "qwen3-vl-flash"  # Fast, cheap, great for captioning
    PLUS = "qwen3-vl-plus"    # Powerful, supports thinking mode
    TURBO = "qwen-turbo"      # Text-only, cheapest for metadata tasks

@dataclass
class AppConfig:
    voyage: VoyageConfig = field(default_factory=VoyageConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    handoff: HandoffConfig = field(default_factory=HandoffConfig)
    ux: UXConfig = field(default_factory=UXConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

try:
    config = AppConfig()
except Exception as e:
    print(f"⚠️ Configuration Warning: {e}")
    config = None