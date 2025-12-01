"""
Escalation Decision Engine
Determines when to hand off to human agents.

Enhanced with multi-turn conversation tracking and API-based token counting.
"""
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from config import config
from llm_client import qwen_client


class EscalationReason(Enum):
    """Why escalation was triggered."""
    USER_REQUEST = "user_requested_human"
    LOW_CONFIDENCE = "low_retrieval_confidence"
    REPEATED_FAILURE = "multiple_failed_attempts"
    NEGATIVE_SENTIMENT = "user_frustration_detected"
    NO_KNOWLEDGE = "outside_knowledge_base"
    COMPLEX_QUERY = "query_too_complex"
    SENSITIVE_TOPIC = "sensitive_topic_detected"


@dataclass
class EscalationDecision:
    """Result of escalation evaluation."""
    should_escalate: bool
    reason: Optional[EscalationReason] = None
    confidence: float = 0.0
    sentiment_score: float = 0.0
    message_to_user: str = ""
    priority: str = "normal"  # "low", "normal", "high", "urgent"


@dataclass
class ConversationContext:
    """
    Enhanced conversation tracking with multimodal support and token management.

    Supports:
    - Multi-turn conversations with turn tracking
    - Image retention logic (keep images for 3 turns)
    - API-based token counting (using actual Qwen API responses)
    - Preemptive pruning before over-budget API calls
    - Context warnings when approaching limits
    """
    conversation_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    failed_attempts: int = 0
    escalation_offered: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    user_metadata: Dict[str, Any] = field(default_factory=dict)

    # New fields for multi-turn support
    current_turn: int = 0
    total_tokens: int = 0
    image_turns: Dict[int, str] = field(default_factory=dict)  # {turn: image_url}
    pruning_notice_shown: bool = False

    # Constants - Updated for Qwen-VL-Plus 32k context window
    WARNING_THRESHOLD: int = 24000  # Warn at 24k tokens
    MAX_TOKENS: int = 28000  # Hard limit at 28k (safety buffer below 32k)
    IMAGE_RETENTION_TURNS: int = 3

    def add_message(self, role: str, content: str):
        """Add a message to history (legacy compatibility)."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })

    def add_user_message_v2(
        self,
        text: str,
        image_url: Optional[str] = None,
        image_filename: Optional[str] = None
    ):
        """
        Add user message with full metadata.

        CRITICAL: Token counting is DEFERRED until API response.
        We don't estimate tokens here - wait for actual usage from Qwen API.
        """
        self.current_turn += 1

        content = []
        metadata = {"has_image": False}

        # Add image if provided
        if image_url:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
            # Track image for retention logic
            self.image_turns[self.current_turn] = image_url
            metadata["has_image"] = True
            metadata["image_name"] = image_filename or "uploaded_image.jpg"

        # Always add text
        content.append({
            "type": "text",
            "text": text
        })

        message = {
            "role": "user",
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "turn": self.current_turn,
            "metadata": metadata
        }

        self.messages.append(message)

        # Token counting moved to add_assistant_message_v2() after API call

    def add_assistant_message_v2(
        self,
        content: str,
        api_usage: Dict[str, int],
        reasoning_content: Optional[str] = None  # Not stored per user requirement
    ):
        """
        Add assistant response with Aeris persona.

        Args:
            content: Assistant response text
            api_usage: Token usage from Qwen API response.usage
                {
                    "prompt_tokens": 1523,
                    "completion_tokens": 89,
                    "total_tokens": 1612
                }
            reasoning_content: NOT stored in history per user requirement

        CRITICAL: Uses ACTUAL token count from API, not character-based estimates.
        """
        message = {
            "role": "assistant",
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "turn": self.current_turn
        }

        self.messages.append(message)

        # Update token count from ACTUAL API usage
        self.total_tokens = api_usage.get("total_tokens", 0)

        # Check if pruning needed
        if self.total_tokens > self.MAX_TOKENS:
            self._auto_prune()

    def should_prune_preemptively(self, num_active_images: int = 0) -> bool:
        """
        Check if we should prune BEFORE making API call.

        Uses conservative heuristics to avoid over-budget API calls:
        1. Message count (>20 message pairs = likely over budget)
        2. Known image costs (num_images * 1800 tokens)
        3. Current total_tokens if available

        This prevents paying for API calls that will exceed limits.
        """
        # Heuristic 1: Too many messages
        if len(self.messages) > 40:  # 20 user-assistant pairs
            return True

        # Heuristic 2: Current tokens + estimated image cost approaching limit
        if self.total_tokens > 0:  # We have token data from previous call
            estimated_image_cost = num_active_images * 1800  # Conservative estimate
            estimated_response_cost = 500  # Average response length
            estimated_total = self.total_tokens + estimated_image_cost + estimated_response_cost

            if estimated_total > (self.MAX_TOKENS * 0.85):  # 85% threshold
                return True

        return False

    def _auto_prune(self):
        """
        Silently prune oldest messages when over token limit.
        User Requirement: Auto-prune silently, show subtle notice.

        CRITICAL: After pruning, we CANNOT accurately recalculate total_tokens
        without re-calling the API. We prune messages and set a conservative estimate.
        The next API call will give us the accurate token count.
        """
        pruned_count = 0

        # Keep pruning oldest message pairs (user + assistant) until we have breathing room
        while len(self.messages) > 4 and self.total_tokens > self.MAX_TOKENS:
            # Remove oldest user-assistant pair
            self.messages.pop(0)  # Remove user message
            if len(self.messages) > 0:
                self.messages.pop(0)  # Remove assistant response
            pruned_count += 1

        # Conservative estimate: assume we've removed enough
        # The next API call will give us accurate count
        if pruned_count > 0:
            # Estimate: each turn pair removed ~1500-3000 tokens
            estimated_removed = pruned_count * 2000
            self.total_tokens = max(0, self.total_tokens - estimated_removed)
            self.pruning_notice_shown = True
            print(f"[ConversationContext] Auto-pruned {pruned_count} message pairs. Next API call will recalibrate token count.")

    def should_include_image_from_turn(self, image_turn: int) -> bool:
        """
        Determine if image from a previous turn should be included in VLM call.
        User Requirement: Pass image for next 2-3 turns only.
        """
        turn_difference = self.current_turn - image_turn
        return turn_difference <= self.IMAGE_RETENTION_TURNS

    def get_messages_for_llm_v2(self) -> List[Dict[str, Any]]:
        """
        Get conversation history for LLM with full metadata.
        User Requirement: Include turn numbers and image markers.
        """
        formatted_messages = []

        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            turn = msg.get("turn", 0)
            metadata = msg.get("metadata", {})

            # Build metadata prefix
            prefix_parts = [f"[Turn {turn}]"]
            if metadata.get("has_image"):
                image_name = metadata.get("image_name", "image")
                prefix_parts.append(f"[📷 User uploaded: {image_name}]")

            prefix = " ".join(prefix_parts)

            # Format content
            if role == "user" and isinstance(content, list):
                # Add prefix to text content
                formatted_content = []
                for item in content:
                    if item.get("type") == "text":
                        formatted_content.append({
                            "type": "text",
                            "text": f"{prefix}\n{item['text']}"
                        })
                    else:
                        formatted_content.append(item)

                formatted_messages.append({
                    "role": role,
                    "content": formatted_content
                })
            else:
                # Assistant message - just pass through
                formatted_messages.append({
                    "role": role,
                    "content": content
                })

        return formatted_messages

    def get_active_images(self, max_images: int = 2) -> List[str]:
        """
        Get image URLs that should be included in current VLM call.
        Only returns images within IMAGE_RETENTION_TURNS.

        CRITICAL: Limits to max_images to control token budget.
        Each image consumes ~1800 tokens (1500-2500 range).
        2 images = 3600 tokens, leaving ~24k for conversation text.

        Args:
            max_images: Maximum number of images to include (default: 2)

        Returns:
            List of image URLs, newest first, up to max_images
        """
        active_images = []
        # Sort by turn (newest first) to prioritize recent images
        for turn in sorted(self.image_turns.keys(), reverse=True):
            if self.should_include_image_from_turn(turn):
                active_images.append(self.image_turns[turn])
                if len(active_images) >= max_images:
                    break  # Stop at budget limit
        return active_images

    def needs_query_rewrite(self, query: str) -> bool:
        """
        Check if query needs rewriting.
        User Requirement: Only if pronouns detected.
        """
        if len(self.messages) < 2:  # No history yet
            return False

        # Check for pronouns
        pronouns = r'\b(it|that|them|these|those|this|its|their)\b'
        return bool(re.search(pronouns, query.lower()))

    def get_warning_message(self) -> Optional[str]:
        """
        Get context limit warning if approaching threshold.
        User Requirement: Warn when approaching, but auto-prune at limit.
        """
        if self.total_tokens > self.WARNING_THRESHOLD and self.total_tokens <= self.MAX_TOKENS:
            return "⚠️ Approaching context limit. Older messages may be hidden soon to manage conversation length."

        if self.pruning_notice_shown:
            self.pruning_notice_shown = False  # Show once
            return "ℹ️ Earlier messages hidden to manage context. Your recent conversation is preserved."

        return None

    def increment_failures(self):
        """Track failed retrieval/response."""
        self.failed_attempts += 1

    def get_recent_messages(self, n: int = 5) -> List[Dict]:
        """Get last n messages."""
        return self.messages[-n:]


class EscalationEngine:
    """
    Evaluates when conversations should escalate to human agents.
    
    Triggers:
    1. Explicit user request ("talk to human")
    2. Low retrieval confidence
    3. LLM admits lack of knowledge
    4. Repeated failures
    5. Negative user sentiment
    """
    
    def __init__(self):
        self.escalation_config = config.escalation
    
    async def evaluate(
        self,
        user_message: str,
        retrieval_confidence: float,
        llm_response: str,
        context: ConversationContext
    ) -> EscalationDecision:
        """
        Evaluate if conversation should escalate.
        
        Args:
            user_message: Latest user message
            retrieval_confidence: Confidence from retrieval
            llm_response: LLM's response content
            context: Conversation context
        
        Returns:
            EscalationDecision
        """
        # === RULE 1: Explicit user request (HIGHEST PRIORITY) ===
        if self._check_explicit_request(user_message):
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.USER_REQUEST,
                confidence=retrieval_confidence,
                message_to_user="I'm connecting you with a human agent now. They'll have our full conversation history.",
                priority="high"
            )
        
        # === RULE 2: LLM admits lack of knowledge ===
        if self._check_uncertainty_response(llm_response):
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.NO_KNOWLEDGE,
                confidence=retrieval_confidence,
                message_to_user="I couldn't find this in my knowledge base. Let me connect you with someone who can help.",
                priority="normal"
            )
        
        # === RULE 3: Very low confidence ===
        if retrieval_confidence < self.escalation_config.low_confidence_threshold:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.LOW_CONFIDENCE,
                confidence=retrieval_confidence,
                message_to_user="I'm not fully confident in my answer. Would you like to speak with a human agent?",
                priority="normal"
            )
        
        # === RULE 4: Repeated failures ===
        if context.failed_attempts >= self.escalation_config.max_failed_attempts:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.REPEATED_FAILURE,
                confidence=retrieval_confidence,
                message_to_user="I'm having trouble helping with this. Let me get a human agent for you.",
                priority="high"
            )
        
        # === RULE 5: Negative sentiment (async) ===
        sentiment_score = await qwen_client.analyze_sentiment(user_message)
        
        if sentiment_score < self.escalation_config.negative_sentiment_threshold:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.NEGATIVE_SENTIMENT,
                confidence=retrieval_confidence,
                sentiment_score=sentiment_score,
                message_to_user="I sense this isn't going well. Would you prefer to speak with a human?",
                priority="high"
            )
        
        # === RULE 6: Warning zone (offer but don't force) ===
        if retrieval_confidence < self.escalation_config.warn_confidence_threshold:
            if not context.escalation_offered:
                context.escalation_offered = True
                return EscalationDecision(
                    should_escalate=False,
                    reason=None,
                    confidence=retrieval_confidence,
                    sentiment_score=sentiment_score,
                    message_to_user="",  # Soft offer appended to response
                    priority="normal"
                )
        
        # === NO ESCALATION NEEDED ===
        return EscalationDecision(
            should_escalate=False,
            reason=None,
            confidence=retrieval_confidence,
            sentiment_score=sentiment_score,
            message_to_user="",
            priority="normal"
        )
    
    def _check_explicit_request(self, message: str) -> bool:
        """Check if user explicitly asked for human."""
        message_lower = message.lower()
        for phrase in self.escalation_config.escalation_phrases:
            if phrase in message_lower:
                return True
        return False
    
    def _check_uncertainty_response(self, response: str) -> bool:
        """Check if LLM response indicates uncertainty."""
        response_lower = response.lower()
        for phrase in self.escalation_config.uncertainty_phrases:
            if phrase in response_lower:
                return True
        return False
    
    async def evaluate_quick(
        self,
        user_message: str,
        retrieval_confidence: float
    ) -> Optional[EscalationDecision]:
        """
        Quick pre-check without LLM response.
        Use before generation to avoid wasted API calls.
        
        Returns:
            EscalationDecision if should escalate immediately, None otherwise
        """
        # Check explicit request
        if self._check_explicit_request(user_message):
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.USER_REQUEST,
                confidence=retrieval_confidence,
                message_to_user="I'm connecting you with a human agent now. They'll have our full conversation history.",
                priority="high"
            )
        
        # Check very low confidence
        if retrieval_confidence < 0.3:
            return EscalationDecision(
                should_escalate=True,
                reason=EscalationReason.LOW_CONFIDENCE,
                confidence=retrieval_confidence,
                message_to_user="I don't have relevant information for this question. Let me connect you with a human agent.",
                priority="normal"
            )
        
        return None


# Singleton instance
escalation_engine = EscalationEngine()
