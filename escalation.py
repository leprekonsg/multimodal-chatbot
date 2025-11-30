"""
Escalation Decision Engine
Determines when to hand off to human agents.
"""
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
    """Tracks conversation state for escalation decisions."""
    conversation_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    failed_attempts: int = 0
    escalation_offered: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    user_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        """Add a message to history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
    
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
