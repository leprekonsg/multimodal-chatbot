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

    # Multi-turn conversation tracking
    turn: int = 0
    conversation_id: str = ""
    context_warning: Optional[str] = None  # Warning message about context limits
    images_retained: int = 0  # Number of user images still in context


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

    def _extract_text_from_content(self, content) -> str:
        """Extract text from potentially multimodal content."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if item.get("type") == "text"
            ]
            return " ".join(text_parts)
        return ""

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
    
    async def chat_stream(
        self,
        message: str,
        image_data: bytes = None,
        conversation_id: str = None,
        user_metadata: Dict = None
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat endpoint with comprehensive error handling.

        Args:
            message: User's text message
            image_data: Optional image bytes from user
            conversation_id: Existing conversation ID
            user_metadata: User info for context

        Yields:
            JSON-formatted messages with tokens and metadata
        """
        import json
        start_time = time.time()

        try:
            # Get conversation context
            context = self.get_or_create_context(conversation_id, user_metadata)
            context.add_message("user", message)

            # Upload user image if provided
            user_image_url = None
            if image_data:
                try:
                    stored = await storage.upload(image_data, "user_upload.jpg")
                    user_image_url = stored.url
                except Exception as e:
                    print(f"⚠️ Failed to upload user image: {e}")

            # Rewrite query if we have conversation history
            search_query = message
            if len(context.messages) > 1:
                try:
                    search_query = await qwen_client.rewrite_query(
                        history=context.messages[:-1],
                        current_query=message
                    )
                except Exception as e:
                    print(f"⚠️ Query rewrite failed: {e}")

            # Step 1: Retrieve relevant documents
            retrieval_result = None
            try:
                retrieval_result = await enhanced_retriever.retrieve(
                    query_text=search_query,
                    query_image=image_data,
                    top_k=config.qdrant.top_k
                )
            except Exception as e:
                print(f"❌ Retrieval failed: {e}")
                traceback.print_exc()
                # Fallback to empty result
                retrieval_result = RetrievalResultV2(documents=[], confidence=0.0, query_intent=None)

            # Build context for generation
            context_text = self._build_context(retrieval_result.documents[:5])
            image_urls = [
                doc.url for doc in retrieval_result.documents
                if doc.type == "image" and doc.url
            ]

            # Stream the response
            async for chunk in self._stream_response(
                query=message,
                context_text=context_text,
                image_urls=image_urls,
                user_image_url=user_image_url,
                retrieval_result=retrieval_result,
                context=context,
                start_time=start_time
            ):
                yield chunk

        except Exception as e:
            # Catastrophic failure - send error and minimal metadata
            print(f"❌ Critical error in chat_stream: {e}")
            traceback.print_exc()

            error_message = "I encountered an error while processing your request. Please try again."
            yield json.dumps({"type": "token", "content": error_message}) + "\n"

            # Send minimal metadata
            latency_ms = (time.time() - start_time) * 1000
            metadata = {
                "type": "metadata",
                "sources": [],
                "source_images": [],
                "confidence": 0.0,
                "latency_ms": latency_ms,
                "conversation_id": conversation_id or "error"
            }
            yield json.dumps(metadata) + "\n"

    async def chat(
        self,
        message: str,
        image_data: bytes = None,
        conversation_id: str = None,
        user_metadata: Dict = None,
        stream: bool = False
    ) -> ChatResponse:
        """
        Main chat endpoint with Aeris persona and multi-turn support.

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

        # Check if this is the first message
        is_first_message = len(context.messages) == 0

        # Upload user image if provided (for VLM context)
        user_image_url = None
        image_filename = None
        if image_data:
            try:
                import time as time_module
                timestamp = int(time_module.time())
                image_filename = f"user_upload_{timestamp}.jpg"
                stored = await storage.upload(image_data, image_filename)
                user_image_url = stored.url
            except Exception as e:
                print(f"⚠️ Failed to upload user image: {e}")
                # Continue without the image rather than crashing

        # Add user message to history using V2 method (with full metadata)
        context.add_user_message_v2(
            text=message,
            image_url=user_image_url,
            image_filename=image_filename
        )

        # Query rewriting: Only if pronouns detected
        # User Requirement: Only if pronouns detected
        search_query = message
        if context.needs_query_rewrite(message):
            try:
                history_text = "\n".join([
                    f"{msg['role'].upper()}: {self._extract_text_from_content(msg['content'])[:200]}"
                    for msg in context.messages[-6:]  # Last 6 messages
                ])

                search_query = await qwen_client.rewrite_query_v2(
                    history_summary=history_text,
                    current_query=message
                )
                print(f"[Query Rewrite] '{message}' → '{search_query}'")
            except Exception as e:
                print(f"⚠️ Query rewrite failed: {e}")
                # Fallback to original query
                search_query = message
        
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

        # Get retrieved image URLs (from knowledge base)
        retrieved_image_urls = [
            doc.url for doc in retrieval_result.documents
            if doc.type == "image" and doc.url
        ]

        # Get active user-uploaded images (within 3-turn window)
        # User Requirement: Pass image for next 2-3 turns only
        # CRITICAL: Limited to 2 images to control token budget (~3600 tokens)
        active_user_images = context.get_active_images(max_images=2)

        # PRE-FLIGHT CHECK: Prune BEFORE expensive API call if needed
        # Prevents over-budget API calls by using heuristics
        if context.should_prune_preemptively(num_active_images=len(active_user_images)):
            context._auto_prune()
            print("[Chatbot] Preemptive pruning to prevent over-budget API call")

        # Step 4: Generate response WITH conversation history
        try:
            # CRITICAL: Limit total images to control token budget
            # Token budget breakdown (28k total):
            # - User images (2 max): ~3600 tokens
            # - KB images (3 max): ~5400 tokens
            # - Total images: ~9000 tokens
            # - Conversation text: ~19000 tokens remaining
            max_kb_images = 3

            vlm_response = await qwen_client.generate_response_v2(
                current_query=message,
                context=context_text,
                retrieved_image_urls=retrieved_image_urls[:max_kb_images],  # Max 3 from KB
                user_uploaded_images=active_user_images,  # Max 2 user images
                conversation_history=context.get_messages_for_llm_v2(),
                enable_thinking=None  # Auto-detect
            )
        except Exception as e:
            print(f"❌ Generation failed: {e}")
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
            latency_ms=latency_ms,
            search_query=search_query  # Pass rewritten query for component filtering
        )

        # Add soft escalation offer if confidence is borderline
        if retrieval_result.confidence < config.escalation.warn_confidence_threshold:
            response.message += "\n\n_If this doesn't fully answer your question, you can ask to speak with a human agent._"

        # Get context warning if any
        context_warning = context.get_warning_message()
        if context_warning:
            response.context_warning = context_warning

        # INLINE WELCOME: Prepend compact greeting on first message
        if is_first_message:
            compact_welcome = "👋 **I'm Aeris, your knowledge assistant.** "
            response.message = compact_welcome + response.message

        # Add assistant message to history WITH actual API token usage
        context.add_assistant_message_v2(
            content=response.message,
            api_usage={
                "prompt_tokens": vlm_response.input_tokens,
                "completion_tokens": vlm_response.output_tokens,
                "total_tokens": vlm_response.input_tokens + vlm_response.output_tokens
            }
        )

        # Add turn and conversation_id to response
        response.turn = context.current_turn
        response.conversation_id = context.conversation_id
        response.images_retained = len(active_user_images)

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
        """Stream response tokens for better perceived latency.

        CRITICAL: Send metadata immediately in first chunk for frontend resilience.
        This ensures conversation_id and turn are available before content streaming.
        """
        import json
        full_response = []

        # EARLY METADATA: Send immediately before streaming content
        # This synchronizes frontend state before any tokens arrive
        try:
            # Pass query for relevant component filtering
            source_images = self._format_source_images(
                retrieval_result.documents[:config.ux.max_sources_displayed],
                query_for_filtering=query
            )
            sources = [
                {
                    "id": doc.id,
                    "type": doc.type,
                    "title": doc.source_display,
                    "url": doc.url,
                    "relevance_score": doc.score
                }
                for doc in retrieval_result.documents[:config.ux.max_sources_displayed]
            ]
        except Exception as e:
            print(f"⚠️ Error formatting sources: {e}")
            source_images = []
            sources = []

        # Send immediate metadata (before tokens) for frontend synchronization
        early_metadata = {
            "type": "metadata",
            "sources": sources,
            "source_images": source_images,
            "confidence": retrieval_result.confidence if retrieval_result else 0.0,
            "latency_ms": 0,
            "conversation_id": context.conversation_id,
            "turn": context.current_turn if hasattr(context, 'current_turn') else 0,
            "context_warning": None  # Will update at end if needed
        }
        yield json.dumps(early_metadata) + "\n"

        try:
            async for token in qwen_client.generate_response_stream(
                query=query,
                context=context_text,
                image_urls=image_urls,
                user_image_url=user_image_url
            ):
                full_response.append(token)
                # Yield token as JSON for consistent parsing
                yield json.dumps({"type": "token", "content": token}) + "\n"
        except Exception as e:
            print(f"❌ Stream error: {e}")
            yield json.dumps({"type": "error", "content": "Error generating response"}) + "\n"

        # After streaming, send final metadata with latency and context warning
        response_text = "".join(full_response)
        if response_text:  # Only save if we got a response
            context.add_message("assistant", response_text)

        latency_ms = (time.time() - start_time) * 1000

        # Get context warning if any (from Phase 1)
        context_warning = context.get_warning_message() if hasattr(context, 'get_warning_message') else None

        # Send FINAL metadata with actual latency and context warning
        # This overwrites the early metadata with complete information
        final_metadata = {
            "type": "metadata",
            "sources": sources,
            "source_images": source_images,
            "confidence": retrieval_result.confidence if retrieval_result else 0.0,
            "latency_ms": latency_ms,
            "conversation_id": context.conversation_id,
            "turn": context.current_turn if hasattr(context, 'current_turn') else 0,
            "context_warning": context_warning
        }
        yield json.dumps(final_metadata) + "\n"
    
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
        latency_ms: float,
        search_query: Optional[str] = None
    ) -> ChatResponse:
        """Build complete response with citations."""
        # Build source list
        sources = []

        for doc in retrieval_result.documents[:config.ux.max_sources_displayed]:
            sources.append(ChatSource(
                id=doc.id,
                type=doc.type,
                title=doc.source_display,
                url=doc.url,
                relevance_score=doc.score
            ))

        # Collect image sources for display with visual grounding data
        # Pass rewritten query for relevant component filtering
        source_images = self._format_source_images(
            retrieval_result.documents[:config.ux.max_sources_displayed],
            query_for_filtering=search_query
        )
        
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
    
    def _filter_relevant_components(
        self,
        components: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Score components based on query relevance using substring containment.

        Returns ALL components with relevance_score field for frontend opacity control.

        Args:
            components: List of component dicts with 'label' and 'bbox_2d'
            query: The query string (preferably rewritten for pronoun resolution)

        Returns:
            All components with 'relevance_score' field (0.0-1.0)

        Implementation:
        - Uses substring containment (naturally handles compound words)
        - "tie" matches "tiedown" without noisy 3-char substrings
        - Returns ALL components with scores (frontend controls display)
        - Logs high-relevance matches for debugging
        """
        import re

        if not components or not query:
            # Return all with default low score
            for comp in components:
                comp['relevance_score'] = 0.2
            return components

        # Only apply scoring if we have many components
        if len(components) <= 10:
            for comp in components:
                comp['relevance_score'] = 0.5  # Medium score for small sets
            return components

        # Stopwords list (common English words to ignore)
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'what', 'which', 'where', 'when', 'who',
            'how', 'why', 'it', 'its', 'their', 'them', 'please', 'show', 'me',
            'find', 'locate', 'i', 'you', 'we'
        }

        # Tokenize query - keep only meaningful tokens
        query_lower = query.lower()

        # Handle hyphenated/compound patterns
        query_lower = re.sub(r'[-_]', ' ', query_lower)

        # Extract tokens
        query_tokens = set(re.findall(r'[a-z0-9]+', query_lower))

        # Remove short tokens and stopwords
        query_tokens = {
            token for token in query_tokens
            if token not in stopwords and len(token) >= 3
        }

        if not query_tokens:
            # No meaningful tokens - score everything low
            for comp in components:
                comp['relevance_score'] = 0.2
            return components

        # Score each component using substring containment
        for comp in components:
            label = comp.get('label', '').lower()
            if not label:
                comp['relevance_score'] = 0.0
                continue

            # Handle hyphenated labels too
            label = re.sub(r'[-_]', ' ', label)

            # Check for substring matches (handles compounds naturally)
            # "tie" in "tiedown" works, but "tie" not in "specified"
            matches = 0
            for token in query_tokens:
                if token in label:
                    matches += 1

            # Calculate relevance as fraction of query tokens matched
            comp['relevance_score'] = matches / len(query_tokens) if query_tokens else 0.0

        # Log results for debugging
        high_relevance = [c for c in components if c['relevance_score'] > 0.25]
        print(f"[Component Filter] {len(high_relevance)}/{len(components)} components with score > 0.25")

        # Show top matches for debugging
        if high_relevance:
            print(f"[Component Filter] Top matches:")
            for comp in sorted(high_relevance, key=lambda x: x['relevance_score'], reverse=True)[:3]:
                print(f"  - {comp.get('label', 'unnamed')[:50]}: {comp['relevance_score']:.2f}")

        return components

    def _format_source_images(
        self,
        documents: List[RetrievedDocumentV2],
        query_for_filtering: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Format source images with visual grounding data for frontend display.

        Args:
            documents: Retrieved documents to format
            query_for_filtering: Optional query string to filter components by relevance

        Returns:
            List of formatted source image dicts with filtered components
        """
        source_images = []

        for doc in documents:
            if doc.type == "image" and doc.url and config.ux.show_source_images:
                # Extract components for visual grounding (bounding boxes)
                components = []
                if doc.metadata:
                    components = doc.metadata.get("components", [])

                # Apply relevance filtering if query provided
                if query_for_filtering:
                    components = self._filter_relevant_components(components, query_for_filtering)

                source_images.append({
                    "url": doc.url,
                    "title": doc.source_display,
                    "caption": doc.caption[:200] if doc.caption else "",
                    "components": components,  # Scored visual grounding data (with relevance_score)
                    "match_type": doc.match_type,
                    "score": doc.score
                })

        return source_images

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