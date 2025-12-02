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
class RerankConfig:
    """
    Voyage Rerank 2 configuration.
    
    Research Evidence:
    - 20-48% accuracy improvement (Databricks)
    - Optimal pipeline: Retrieve 50 → Rerank → Top 5 to LLM
    """
    enabled: bool = True
    model_name: str = "rerank-2"  # Voyage Rerank 2
    top_k: int = 5  # Final results to LLM
    candidates_to_rerank: int = 50  # Candidates from hybrid search
    # Set to False to disable reranking (e.g., for latency-critical use cases)
    # Reranking adds ~300-600ms but significantly improves precision

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

    # Component relevance display thresholds (4-tier visual hierarchy)
    # These define the breakpoints for gradual opacity/styling
    component_relevance_tiers: dict = field(default_factory=lambda: {
        "highly_relevant": 0.75,    # Tier 1: 75-100% match
        "relevant": 0.50,            # Tier 2: 50-74% match
        "somewhat_relevant": 0.25,   # Tier 3: 25-49% match
        "low_relevant": 0.0          # Tier 4: 0-24% match
    })

    # Visual styling for each tier (guidance for frontend)
    component_visual_tiers: dict = field(default_factory=lambda: {
        "highly_relevant": {
            "opacity": 1.0,
            "color": "#2563eb",    # Bright blue
            "border_width": 3,
            "font_size": "13px",
            "font_weight": "700"
        },
        "relevant": {
            "opacity": 0.9,
            "color": "#3b82f6",    # Blue
            "border_width": 2,
            "font_size": "12px",
            "font_weight": "600"
        },
        "somewhat_relevant": {
            "opacity": 0.6,
            "color": "#94a3b8",    # Gray-blue
            "border_width": 1,
            "font_size": "11px",
            "font_weight": "400"
        },
        "low_relevant": {
            "opacity": 0.3,
            "color": "#9ca3af",    # Gray
            "border_width": 1,
            "font_size": "10px",
            "font_weight": "300"
        }
    })

@dataclass
class EmbeddingConfig:
    dimension: int = 1024

@dataclass
class ConversationConfig:
    """Multi-turn conversation settings."""
    # User Requirements - UPDATED for Qwen-VL-Plus 32k context window
    max_context_tokens: int = 28000  # Was 8000 - updated to reflect 32k model capability
    warning_threshold: int = 24000  # Was 6500 - warn earlier to give users heads up
    image_retention_turns: int = 3
    chatbot_name: str = "Aeris"
    welcome_enabled: bool = True
    persona_style: str = "first_person_with_name"  # "I'm Aeris..."

    # Pronoun patterns for query rewrite detection
    pronoun_patterns: List[str] = field(default_factory=lambda: [
        r'\b(it|that|them|these|those|this|its|their)\b'
    ])

    # Token counting strategy
    use_api_token_counts: bool = True  # CRITICAL: Always use actual API response counts

    # Image token budgeting (NEW - addresses IMPLEMENTATION_PLAN_ANALYSIS Issue #3)
    max_user_images: int = 2  # Max user-uploaded images to include (2 = ~3600 tokens)
    max_kb_images: int = 3  # Max knowledge base images to include (3 = ~5400 tokens)
    # Total: 5 images max = ~9000 tokens, leaving 19k for conversation

    # Preemptive pruning (NEW - addresses IMPLEMENTATION_PLAN_ANALYSIS Issue #1)
    enable_preemptive_pruning: bool = True
    preemptive_prune_threshold: float = 0.85  # Prune when at 85% estimated capacity

# UPDATED: Using Qwen3-VL models
class ModelTier(Enum):
    FLASH = "qwen3-vl-flash"  # Fast, cheap, great for captioning
    PLUS = "qwen3-vl-plus"    # Powerful, supports thinking mode
    TURBO = "qwen-turbo"      # Text-only, cheapest for metadata tasks

@dataclass
class AppConfig:
    voyage: VoyageConfig = field(default_factory=VoyageConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    qwen: QwenConfig = field(default_factory=QwenConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    handoff: HandoffConfig = field(default_factory=HandoffConfig)
    ux: UXConfig = field(default_factory=UXConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)

try:
    config = AppConfig()
except Exception as e:
    print(f"âš ï¸ Configuration Warning: {e}")
    config = None