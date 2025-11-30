"""
Human Handoff Integrations
Connects to helpdesk systems for seamless escalation.
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

import httpx

from config import config
from escalation import EscalationReason, ConversationContext


@dataclass
class HandoffResult:
    """Result of handoff operation."""
    success: bool
    ticket_id: Optional[str] = None
    agent_assigned: Optional[str] = None
    estimated_wait: Optional[str] = None
    error: Optional[str] = None


class HandoffProvider(ABC):
    """Base class for handoff providers."""
    
    @abstractmethod
    async def send_handoff(
        self,
        context: ConversationContext,
        reason: EscalationReason,
        summary: str,
        retrieved_context: list
    ) -> HandoffResult:
        """Send conversation to human agent."""
        pass


class WebhookHandoff(HandoffProvider):
    """
    Generic webhook handoff for:
    - Zendesk
    - Freshdesk
    - Intercom
    - Custom helpdesk
    """
    
    def __init__(self):
        self.webhook_url = config.handoff.webhook_url
        self.api_key = config.handoff.webhook_api_key
    
    async def send_handoff(
        self,
        context: ConversationContext,
        reason: EscalationReason,
        summary: str,
        retrieved_context: list
    ) -> HandoffResult:
        """Send to helpdesk via webhook."""
        if not self.webhook_url:
            return HandoffResult(
                success=False,
                error="Webhook URL not configured"
            )
        
        payload = {
            "ticket": {
                "subject": f"Chatbot Escalation: {reason.value}",
                "description": self._build_description(context, reason, summary),
                "priority": self._get_priority(reason),
                "tags": ["chatbot-escalation", reason.value],
                "custom_fields": {
                    "conversation_id": context.conversation_id,
                    "escalation_reason": reason.value,
                    "failed_attempts": context.failed_attempts
                }
            },
            "conversation": {
                "id": context.conversation_id,
                "messages": context.messages,
                "created_at": context.created_at.isoformat()
            },
            "context": {
                "retrieved_documents": [
                    {
                        "id": doc.id,
                        "type": doc.type,
                        "caption": doc.caption if hasattr(doc, 'caption') else None,
                        "url": doc.url if hasattr(doc, 'url') else None
                    }
                    for doc in retrieved_context
                ],
                "summary": summary
            },
            "user": context.user_metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code in (200, 201):
                    data = response.json()
                    return HandoffResult(
                        success=True,
                        ticket_id=data.get("ticket_id") or data.get("id"),
                        agent_assigned=data.get("agent"),
                        estimated_wait=data.get("estimated_wait")
                    )
                else:
                    return HandoffResult(
                        success=False,
                        error=f"Webhook returned {response.status_code}"
                    )
        
        except Exception as e:
            return HandoffResult(
                success=False,
                error=str(e)
            )
    
    def _build_description(
        self,
        context: ConversationContext,
        reason: EscalationReason,
        summary: str
    ) -> str:
        """Build ticket description."""
        recent_messages = context.get_recent_messages(5)
        message_text = "\n".join([
            f"[{m['role']}]: {m['content'][:200]}"
            for m in recent_messages
        ])
        
        return f"""Escalated from AI Chatbot

Reason: {reason.value}

Summary:
{summary}

Recent Conversation:
{message_text}

---
Conversation ID: {context.conversation_id}
Failed Attempts: {context.failed_attempts}
"""
    
    def _get_priority(self, reason: EscalationReason) -> str:
        """Map escalation reason to ticket priority."""
        high_priority = [
            EscalationReason.NEGATIVE_SENTIMENT,
            EscalationReason.USER_REQUEST
        ]
        if reason in high_priority:
            return "high"
        return "normal"


class SlackHandoff(HandoffProvider):
    """
    Slack integration for real-time handoff.
    Posts to support channel with claim button.
    """
    
    def __init__(self):
        self.bot_token = config.handoff.slack_bot_token
        self.channel = config.handoff.slack_channel
        self.app_base_url = config.handoff.app_base_url
    
    async def send_handoff(
        self,
        context: ConversationContext,
        reason: EscalationReason,
        summary: str,
        retrieved_context: list
    ) -> HandoffResult:
        """Post escalation to Slack channel."""
        if not self.bot_token or not self.channel:
            return HandoffResult(
                success=False,
                error="Slack not configured"
            )
        
        blocks = self._build_slack_blocks(context, reason, summary)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://slack.com/api/chat.postMessage",
                    json={
                        "channel": self.channel,
                        "blocks": blocks,
                        "text": f"ðŸš¨ Escalation: {reason.value}"
                    },
                    headers={
                        "Authorization": f"Bearer {self.bot_token}",
                        "Content-Type": "application/json"
                    }
                )
                
                data = response.json()
                
                if data.get("ok"):
                    return HandoffResult(
                        success=True,
                        ticket_id=data.get("ts")  # Message timestamp as ID
                    )
                else:
                    return HandoffResult(
                        success=False,
                        error=data.get("error", "Unknown Slack error")
                    )
        
        except Exception as e:
            return HandoffResult(
                success=False,
                error=str(e)
            )
    
    def _build_slack_blocks(
        self,
        context: ConversationContext,
        reason: EscalationReason,
        summary: str
    ) -> list:
        """Build Slack Block Kit message."""
        last_message = context.messages[-1]["content"] if context.messages else "N/A"
        
        # Emoji based on reason
        emoji_map = {
            EscalationReason.USER_REQUEST: "ðŸ™‹",
            EscalationReason.NEGATIVE_SENTIMENT: "ðŸ˜¤",
            EscalationReason.LOW_CONFIDENCE: "â“",
            EscalationReason.NO_KNOWLEDGE: "ðŸ“š",
            EscalationReason.REPEATED_FAILURE: "ðŸ”„"
        }
        emoji = emoji_map.get(reason, "ðŸš¨")
        
        return [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Escalation: {reason.value.replace('_', ' ').title()}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*User Query:*\n{last_message[:200]}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Conversation ID:*\n`{context.conversation_id}`"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary:*\n{summary[:500]}"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Failed attempts: {context.failed_attempts} | Messages: {len(context.messages)}"
                    }
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "âœ‹ Claim This"
                        },
                        "style": "primary",
                        "action_id": "claim_conversation",
                        "value": context.conversation_id
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "View History"
                        },
                        "url": f"{self.app_base_url}/conversations/{context.conversation_id}"
                    }
                ]
            }
        ]


class HandoffManager:
    """
    Manages handoff to human agents.
    Supports multiple providers with fallback.
    """
    
    def __init__(self):
        self.providers: Dict[str, HandoffProvider] = {}
        
        # Initialize available providers
        if config.handoff.webhook_url:
            self.providers["webhook"] = WebhookHandoff()
        
        if config.handoff.slack_bot_token:
            self.providers["slack"] = SlackHandoff()
    
    async def handoff(
        self,
        context: ConversationContext,
        reason: EscalationReason,
        summary: str,
        retrieved_context: list = None,
        preferred_provider: str = None
    ) -> HandoffResult:
        """
        Execute handoff to human agent.
        
        Args:
            context: Conversation context
            reason: Why escalating
            summary: AI-generated summary of issue
            retrieved_context: Documents retrieved during conversation
            preferred_provider: Preferred provider (webhook, slack)
        
        Returns:
            HandoffResult
        """
        retrieved_context = retrieved_context or []
        
        # Use preferred provider if available
        if preferred_provider and preferred_provider in self.providers:
            return await self.providers[preferred_provider].send_handoff(
                context, reason, summary, retrieved_context
            )
        
        # Try all providers in order
        errors = []
        for name, provider in self.providers.items():
            result = await provider.send_handoff(
                context, reason, summary, retrieved_context
            )
            if result.success:
                return result
            errors.append(f"{name}: {result.error}")
        
        # All failed
        if not self.providers:
            return HandoffResult(
                success=False,
                error="No handoff providers configured"
            )
        
        return HandoffResult(
            success=False,
            error=f"All providers failed: {'; '.join(errors)}"
        )
    
    @property
    def available_providers(self) -> list:
        """List available provider names."""
        return list(self.providers.keys())


# Singleton instance
handoff_manager = HandoffManager()
