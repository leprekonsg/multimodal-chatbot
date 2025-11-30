"""
Main Chatbot Module
Orchestrates retrieval, generation, and escalation with UX priority.
"""
import asyncio
import time
import traceback
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from config import config
from retrieval import enhanced_retriever, RetrievalResultV2, RetrievedDocumentV2
from llm_client import qwen_client, VLMResponse
from escalation import escalation_engine, EscalationDecision, ConversationContext, EscalationReason
from handoff import handoff_manager, HandoffResult
from storage import storage


@dataclass
class ChatSource:
    """Source citation for response."""
    id: str
    type: str
    title: str
    url: Optional[str] = None
    relevance_score: float = 0.0


@dataclass
class ChatResponse:
    """Complete response to user query."""
    message: str
    sources: List[ChatSource] = field(default_factory=list)
    confidence: float = 0.0
    escalated: bool = False
    escalation_reason: Optional[str] = None
    thinking_enabled: bool = False
    latency_ms: float = 0.0
    
    # For UX - display source images
    source_images: List[Dict[str, str]] = field(default_factory=list)


class MultimodalChatbot:
    """
    Production-ready multimodal RAG chatbot.
    
    UX Priorities:
    1. Fast responses (target <2s TTFT)
    2. Clear source citations
    3. Smooth escalation when needed
    4. Streaming support
    """
    
    def __init__(self):
        self.conversations: Dict[str, ConversationContext] = {}
    
    def get_or_create_context(
        self,
        conversation_id: str = None,
        user_metadata: Dict = None
    ) -> ConversationContext:
        """Get existing or create new conversation context."""
        if conversation_id and conversation_id in self.conversations:
            return self.conversations[conversation_id]
        
        conv_id = conversation_id or str(uuid.uuid4())
        context = ConversationContext(
            conversation_id=conv_id,
            user_metadata=user_metadata or {}
        )
        self.conversations[conv_id] = context
        return context
    
    async def chat(
        self,
        message: str,
        image_data: bytes = None,
        conversation_id: str = None,
        user_metadata: Dict = None,
        stream: bool = False
    ) -> ChatResponse:
        """
        Main chat endpoint.
        
        Args:
            message: User's text message
            image_data: Optional image bytes from user
            conversation_id: Existing conversation ID
            user_metadata: User info for context
            stream: Return streaming response
        
        Returns:
            ChatResponse with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Get conversation context
        context = self.get_or_create_context(conversation_id, user_metadata)
        context.add_message("user", message)
        
        # Upload user image if provided (for VLM context)
        user_image_url = None
        if image_data:
            try:
                stored = await storage.upload(image_data, "user_upload.jpg")
                user_image_url = stored.url
            except Exception as e:
                print(f"Ã¢Å¡Â Ã¯Â¸Â Failed to upload user image: {e}")
                # Continue without the image rather than crashing
        
        # Rewrite query if we have conversation history
        search_query = message
        if len(context.messages) > 1:
            try:
                search_query = await qwen_client.rewrite_query(
                    history=context.messages[:-1],
                    current_query=message
                )
            except Exception as e:
                print(f"Ã¢Å¡Â Ã¯Â¸Â Query rewrite failed: {e}")
        
        # Step 1: Retrieve relevant documents (Using V2 Pipeline)
        # Note: We pass raw bytes to retrieval for perceptual hashing
        try:
            retrieval_result: RetrievalResultV2 = await enhanced_retriever.retrieve(
                query_text=search_query,
                query_image=image_data,
                top_k=config.qdrant.top_k
            )
        except Exception as e:
            print(f"Ã¢ÂÅ’ Retrieval failed: {e}")
            traceback.print_exc()
            # Fallback to empty result to allow graceful failure or escalation
            retrieval_result = RetrievalResultV2(documents=[], confidence=0.0, query_intent=None)
        
        # Step 2: Quick escalation check (before generation)
        quick_decision = await escalation_engine.evaluate_quick(
            user_message=message,
            retrieval_confidence=retrieval_result.confidence
        )
        
        if quick_decision and quick_decision.should_escalate:
            print(f"[Chatbot] Quick escalation triggered: {quick_decision.reason}")
            print(f"[Chatbot] Confidence was: {retrieval_result.confidence:.2%}")
            return await self._handle_escalation(
                context=context,
                decision=quick_decision,
                retrieved_docs=retrieval_result.documents,
                start_time=start_time
            )
        
        # Step 3: Build context for generation
        context_text = self._build_context(retrieval_result.documents)
        print(f"[Chatbot] Built context from {len(retrieval_result.documents)} documents")
        
        # Extract image URLs from retrieved docs for the VLM
        image_urls = [
            doc.url for doc in retrieval_result.documents
            if doc.type == "image" and doc.url
        ]
        
        # Step 4: Generate response
        if stream:
            # Return the async generator directly
            return self._stream_response(
                query=message,
                context_text=context_text,
                image_urls=image_urls,
                user_image_url=user_image_url,
                retrieval_result=retrieval_result,
                context=context,
                start_time=start_time
            )
        
        try:
            vlm_response = await qwen_client.generate_response(
                query=message,
                context=context_text,
                image_urls=image_urls,
                user_image_url=user_image_url
            )
        except Exception as e:
            print(f"Ã¢ÂÅ’ Generation failed: {e}")
            return ChatResponse(
                message="I'm having trouble generating a response right now. Please try again.",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000
            )
        
        # Step 5: Full escalation evaluation
        decision = await escalation_engine.evaluate(
            user_message=message,
            retrieval_confidence=retrieval_result.confidence,
            llm_response=vlm_response.content,
            context=context
        )
        
        if decision.should_escalate:
            return await self._handle_escalation(
                context=context,
                decision=decision,
                retrieved_docs=retrieval_result.documents,
                start_time=start_time,
                partial_response=vlm_response.content
            )
        
        # Step 6: Build response with sources
        latency_ms = (time.time() - start_time) * 1000
        
        response = self._build_response(
            content=vlm_response.content,
            retrieval_result=retrieval_result,
            vlm_response=vlm_response,
            latency_ms=latency_ms
        )
        
        # Add soft escalation offer if confidence is borderline
        if retrieval_result.confidence < config.escalation.warn_confidence_threshold:
            response.message += "\n\n_If this doesn't fully answer your question, you can ask to speak with a human agent._"
        
        context.add_message("assistant", response.message)
        
        return response
    
    async def _stream_response(
        self,
        query: str,
        context_text: str,
        image_urls: List[str],
        user_image_url: str,
        retrieval_result: RetrievalResultV2,
        context: ConversationContext,
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens for better perceived latency."""
        full_response = []
        
        try:
            async for token in qwen_client.generate_response_stream(
                query=query,
                context=context_text,
                image_urls=image_urls,
                user_image_url=user_image_url
            ):
                full_response.append(token)
                yield token
        except Exception as e:
            print(f"Ã¢ÂÅ’ Stream error: {e}")
            yield "\n[Error generating response]"
        
        # After streaming, add sources
        response_text = "".join(full_response)
        sources_text = self._format_sources(retrieval_result.documents[:3])
        
        if sources_text:
            yield f"\n\n{sources_text}"
        
        context.add_message("assistant", response_text)
    
    async def _handle_escalation(
        self,
        context: ConversationContext,
        decision: EscalationDecision,
        retrieved_docs: List[RetrievedDocumentV2],
        start_time: float,
        partial_response: str = None
    ) -> ChatResponse:
        """Handle escalation to human agent."""
        
        # Generate summary for agent
        summary = f"User asked: {context.messages[-1]['content'][:200]}"
        if partial_response:
            summary += f"\n\nBot attempted response: {partial_response[:300]}"
        
        # Execute handoff
        handoff_result = await handoff_manager.handoff(
            context=context,
            reason=decision.reason,
            summary=summary,
            retrieved_context=retrieved_docs
        )
        
        # Build escalation message
        if handoff_result.success:
            message = decision.message_to_user
            if handoff_result.estimated_wait:
                message += f" Estimated wait: {handoff_result.estimated_wait}."
            if handoff_result.ticket_id:
                message += f" (Reference: #{handoff_result.ticket_id})"
        else:
            message = decision.message_to_user + " (Our team will follow up shortly.)"
        
        latency_ms = (time.time() - start_time) * 1000
        
        context.add_message("assistant", message)
        
        return ChatResponse(
            message=message,
            sources=[],
            confidence=decision.confidence,
            escalated=True,
            escalation_reason=decision.reason.value if decision.reason else None,
            latency_ms=latency_ms
        )
    
    def _build_context(self, documents: List[RetrievedDocumentV2]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(documents[:5]):
            # Add match type info to help LLM understand why this was retrieved
            match_info = ""
            if doc.is_exact_match:
                match_info = " [EXACT IMAGE MATCH]"
            elif doc.match_type == "visual":
                match_info = " [VISUALLY SIMILAR]"
            
            if doc.type == "image":
                context_parts.append(
                    f"[Image {i+1}: {doc.source_display}]{match_info}\n{doc.caption}"
                )
            else:
                context_parts.append(
                    f"[Document {i+1}: {doc.source_display}]\n{doc.text[:1000]}"
                )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_response(
        self,
        content: str,
        retrieval_result: RetrievalResultV2,
        vlm_response: VLMResponse,
        latency_ms: float
    ) -> ChatResponse:
        """Build complete response with citations."""
        # Build source list
        sources = []
        source_images = []
        
        for doc in retrieval_result.documents[:config.ux.max_sources_displayed]:
            sources.append(ChatSource(
                id=doc.id,
                type=doc.type,
                title=doc.source_display,
                url=doc.url,
                relevance_score=doc.score
            ))
            
            # Collect image sources for display with visual grounding data
            if doc.type == "image" and doc.url and config.ux.show_source_images:
                # Extract components for visual grounding (bounding boxes)
                components = []
                if doc.metadata:
                    components = doc.metadata.get("components", [])
                
                source_images.append({
                    "url": doc.url,
                    "title": doc.source_display,
                    "caption": doc.caption[:200] if doc.caption else "",
                    "components": components,  # Visual grounding data
                    "match_type": doc.match_type,
                    "score": doc.score
                })
        
        # Don't append sources to message - they're shown in the sidebar
        # This avoids the duplicate/broken markdown link issue
        message = content
        
        return ChatResponse(
            message=message,
            sources=sources,
            confidence=retrieval_result.confidence,
            escalated=False,
            thinking_enabled=vlm_response.thinking_enabled,
            latency_ms=latency_ms,
            source_images=source_images
        )
    
    def _format_sources(self, documents: List[RetrievedDocumentV2]) -> str:
        """Format source citations for display."""
        if not documents:
            return ""
        
        lines = ["\n\n**Sources:**"]
        for i, doc in enumerate(documents[:3]):
            if doc.type == "image":
                lines.append(f"- [{doc.source_display}]({doc.url})" if doc.url else f"- {doc.source_display}")
            else:
                lines.append(f"- {doc.source_display}")
        
        return "\n".join(lines)
    
    def clear_conversation(self, conversation_id: str):
        """Clear a conversation from memory."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]


# Singleton instance
chatbot = MultimodalChatbot()