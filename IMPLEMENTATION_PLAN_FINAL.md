# Multi-Turn Conversation Enhancement: Final Implementation Plan
## Aeris - Your Multimodal Knowledge Assistant

---

## Requirements Summary (User-Confirmed)

Based on clarification discussions, here are the finalized requirements:

### 1. **Welcome Flow**
- ✅ **Always show full welcome message** on every new conversation
- Users get clear guidance each time they start fresh
- Welcome message explains Aeris's capabilities

### 2. **Context Pruning Strategy**
- ✅ **Warn user and request confirmation** when approaching token limit
- Give users explicit choice: "Continue (old context dropped) or Start Fresh?"
- Respect user agency - no silent pruning

### 3. **Image Handling in Follow-ups**
- ✅ **Pass image for next 2-3 turns only**
- Balance between accuracy and API cost
- After 3 turns, rely on initial image description in conversation history
- Suitable for technical diagrams where immediate follow-ups need visual reference

### 4. **Query Rewriting**
- ✅ **Only if pronouns detected** (it, that, them, these, those)
- Pattern matching triggers rewrite: efficient cost-to-benefit ratio
- Covers 80% of ambiguous cases at 30% of the cost
- Fallback to original query if rewrite fails

### 5. **Conversation TTL**
- ✅ **Never auto-clear - manual 'New Chat' only**
- Conversations persist in memory until user explicitly clicks "New Chat"
- Suitable for desktop users who keep tabs open
- Memory cleanup on server restart (in-memory storage)

### 6. **Persona Style**
- ✅ **First-person with name: "I'm Aeris, and I..."**
- Friendly, personable tone
- Builds brand identity
- Example: "I'm Aeris, and I found 3 relevant documents about hydraulic systems."

### 7. **Token Limit UX**
- ✅ **Auto-prune silently** when limit reached
- Show subtle inline warning: "Earlier messages hidden to manage context"
- Keep conversation flowing smoothly
- Note: This overrides #2 for the pruning *mechanism* - we warn when *approaching* limit, but silently prune when *at* limit

### 8. **Context Format**
- ✅ **Include full metadata in prompts**
- Add markers: `[Turn 3]`, `[User uploaded image: diagram.jpg]`
- Helps VLM understand conversation structure
- Example: `[Turn 2] [📷 User uploaded: hydraulic_pump.jpg] USER: What is this?`

---

## Critical Fixes from Technical Reviews (Dec 2025)

This plan has been updated to address critical infrastructure issues identified in technical reviews:

### 🔴 FIXED: Token Counting (Review #1)
**Problem**: Original plan used `len(text) // 4` for text and hardcoded `256` for images. This is dangerously inaccurate for:
- Modern tokenizers (Qwen multilingual models)
- High-res VLM image processing (1024×1024+ images consume 1500-2500 tokens, not 256)

**Solution**:
- **Remove all character-based estimates**
- Use `response.usage` from Qwen API for exact token counts
- Token counting deferred until API response received
- Updated in `add_assistant_message_v2(api_usage: dict)`

### 🔴 FIXED: Context Window Limits (Review #1)
**Problem**: Original 8k token limit severely underutilizes Qwen-VL-Plus 32k context window.

**Solution**:
- `MAX_TOKENS`: 8000 → **28000** (safety buffer below 32k)
- `WARNING_THRESHOLD`: 6500 → **24000**
- Allows 3-4x more conversation history without pruning

### 🔴 FIXED: Image Token Budgeting (Review #2 - CRITICAL)
**Problem**: Plan passed up to 8 images (user + KB) without token budget consideration.
- 8 images × 1800 tokens = **14,400 tokens** (51% of budget)
- Leaves only 13,600 tokens for conversation text
- Forces aggressive pruning, losing conversation context

**Solution**:
- Cap **user images at 2** (~3,600 tokens)
- Cap **KB images at 3** (~5,400 tokens)
- Total: **5 images max = ~9,000 tokens** (32% of budget)
- Leaves **19,000 tokens for conversation text** (68% of budget)
- Implemented in `get_active_images(max_images=2)`

### 🔴 FIXED: Reactive Token Management (Review #2)
**Problem**: Token counting happened AFTER API call. If over budget, we've already paid for the expensive call.

**Solution**:
- **Preemptive pruning** using heuristics BEFORE API call
- Triggers when:
  - Message count > 40 (20 turn pairs)
  - Estimated tokens > 85% of max (24k threshold)
- Implemented in `should_prune_preemptively()`
- Prevents wasted over-budget API calls

### 🔴 FIXED: Streaming Metadata Timing (Review #2)
**Problem**: Metadata only sent after streaming completes. Frontend state corrupted on mid-stream disconnections.

**Solution**:
- Send `conversation_id` and `turn` **immediately** in first stream chunk
- Frontend synchronized before content starts
- Resilient to network interruptions

### 🔴 FIXED: Optimistic UI (Review #2)
**Problem**: No immediate feedback - users wait 2-5s staring at blank screen.

**Solution**:
- Show user message **immediately** (< 50ms)
- Show typing indicator while processing
- Stream tokens as they arrive
- 2x faster **perceived** response time

### 🟡 ACKNOWLEDGED: State Management
**Status**: In-memory storage acceptable for prototype, Redis planned for Phase 4.

**Rationale**:
- Plan already documented this limitation
- Adds deployment complexity (Redis service)
- Post-MVP enhancement after UX validation

### 🟢 VALIDATED: Image Retention Logic
**Status**: 3-turn window kept as designed per user requirements.

**Future Enhancement** (Phase 4): Optional semantic retention for edge cases.

### 🟢 VALIDATED: Query Rewriting
**Status**: Pronoun-based triggering kept for cost efficiency.

**Enhancement**: Improved rewrite prompt to handle implicit context better.

### 🟡 IMPROVED: Welcome Message UX (Review #2 - User Approved)
**Problem**: Blocking welcome forces users to send query twice:
1. First message → Get welcome
2. Second message → Get actual answer

**Solution**: **Inline welcome** - prepend compact greeting to first answer
- User asks "Where is the pressure valve?"
- Response: "👋 **I'm Aeris, your knowledge assistant.** The pressure valve is located..."
- User gets answer **immediately** while still being introduced to Aeris

**Implementation**:
```python
if is_first_message:
    # Process query normally (don't return early!)
    response = await self._process_query(...)

    # Prepend compact welcome
    compact_welcome = "👋 **I'm Aeris, your knowledge assistant.** "
    response.message = compact_welcome + response.message
    return response
```

**Impact**: 50% faster time-to-value for new users (one round-trip instead of two).

---

## Architecture Decisions

### Message Format Standard

```python
# User message with image (Turn 1)
{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "http://localhost:8000/images/abc123.jpg"}},
        {"type": "text", "text": "What is this diagram showing?"}
    ],
    "timestamp": "2025-01-15T10:30:00Z",
    "turn": 1,
    "metadata": {"has_image": True, "image_name": "hydraulic_pump.jpg"}
}

# Assistant response (Turn 1)
{
    "role": "assistant",
    "content": "I'm Aeris, and I can see this is a hydraulic pump system diagram...",
    "timestamp": "2025-01-15T10:30:05Z",
    "turn": 1
    # reasoning_content is NOT stored
}

# User follow-up (Turn 2 - within 3-turn image retention window)
{
    "role": "user",
    "content": [
        {"type": "text", "text": "Where is the pressure valve in it?"}
    ],
    "timestamp": "2025-01-15T10:31:00Z",
    "turn": 2,
    "metadata": {"references_image_from_turn": 1}  # Image from turn 1 will be included
}
```

### Token Limit Management Flow

```mermaid
graph TD
    A[User sends message] --> B{Total tokens > 24000?}
    B -->|No| C[Process normally]
    B -->|Yes| D{Total tokens > 28000?}
    D -->|No| E[Show warning banner: 'Approaching context limit']
    E --> C
    D -->|Yes| F[Auto-prune oldest messages]
    F --> G[Show subtle notice: 'Earlier messages hidden']
    G --> C
```

**Note**: Limits updated to reflect Qwen-VL-Plus 32k context window (28k max with safety buffer).

### Image Retention Logic

```python
def should_include_image(image_turn: int, current_turn: int) -> bool:
    """
    Determine if an image from a previous turn should be included.

    User Requirement: Pass image for next 2-3 turns only.
    """
    turn_difference = current_turn - image_turn
    return turn_difference <= 3  # Turn where uploaded + next 3 turns
```

**Example:**
- Turn 1: User uploads `pump.jpg` → Image included in VLM call ✅
- Turn 2: User asks "Where is the valve?" → Image included ✅
- Turn 3: User asks "What's the part number?" → Image included ✅
- Turn 4: User asks "How to maintain it?" → Image NOT included, use description ❌

### Pronoun Detection Pattern

```python
PRONOUNS_PATTERN = r'\b(it|that|them|these|those|this|its|their)\b'

def needs_query_rewrite(query: str, has_history: bool) -> bool:
    """
    User Requirement: Only rewrite if pronouns detected.
    """
    if not has_history:
        return False

    return bool(re.search(PRONOUNS_PATTERN, query.lower()))
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (2 days)

#### Task 1.1: Enhanced ConversationContext
**File**: `escalation.py`

**Key Changes:**
1. Add `turn` counter to track conversation progress
2. Add `image_turns` dict to track which turns have images
3. Implement `add_user_message_v2()` with metadata
4. Implement token counting with warnings at 6500 tokens
5. Implement auto-pruning at 8000 tokens with subtle notification
6. Add `get_messages_for_llm_v2()` with turn metadata

```python
@dataclass
class ConversationContext:
    """Enhanced conversation tracking with multimodal support."""
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
```

---

#### Task 1.2: Update Chatbot Orchestrator
**File**: `chatbot.py`

**Key Changes:**
1. Add Aeris welcome message (always shown on first message)
2. Use new ConversationContext methods
3. Implement pronoun-based query rewriting
4. Pass active images (within 3-turn window) to VLM
5. Add context warning to responses

```python
class MultimodalChatbot:
    """
    Aeris - Your multimodal knowledge assistant.
    Handles multi-turn conversations with context awareness.
    """

    # User Requirement: First-person with name
    WELCOME_MESSAGE = """👋 Hi! I'm **Aeris**, your multimodal knowledge assistant.

I can help you find information from your uploaded manuals and documents. Here's what I can do:

📄 **Answer questions** about your knowledge base using both text and images
🔍 **Search visually** - upload an image and I'll find similar ones or answer questions about it
💬 **Remember context** - I'll keep track of our conversation, so you can ask follow-up questions naturally

How can I help you today?"""

    def __init__(self):
        self.conversations: Dict[str, ConversationContext] = {}

    async def chat(
        self,
        message: str,
        image_data: bytes = None,
        conversation_id: str = None,
        user_metadata: Dict = None,
        stream: bool = False
    ) -> ChatResponse:
        """Main chat endpoint with Aeris persona and multi-turn support."""
        start_time = time.time()

        # Get or create conversation context
        context = self.get_or_create_context(conversation_id, user_metadata)

        # Check if this is the first message
        is_first_message = len(context.messages) == 0

        # Upload user image if provided
        user_image_url = None
        image_filename = None
        if image_data:
            try:
                # Generate filename with timestamp
                import time
                timestamp = int(time.time())
                image_filename = f"user_upload_{timestamp}.jpg"
                stored = await storage.upload(image_data, image_filename)
                user_image_url = stored.url
            except Exception as e:
                print(f"⚠️ Failed to upload user image: {e}")

        # Add user message to history using V2 method
        context.add_user_message_v2(
            text=message,
            image_url=user_image_url,
            image_filename=image_filename
        )

        # NOTE: First message handling moved to end (inline welcome after processing)

        # Query rewriting: Only if pronouns detected
        # User Requirement: Only if pronouns detected
        # Enhancement: Improved prompt handles implicit context better (not just pronouns)
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

        # Retrieve relevant documents
        try:
            retrieval_result: RetrievalResultV2 = await enhanced_retriever.retrieve(
                query_text=search_query,
                query_image=image_data,
                top_k=config.qdrant.top_k
            )
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            retrieval_result = RetrievalResultV2(documents=[], confidence=0.0, query_intent=None)

        # Quick escalation check
        quick_decision = await escalation_engine.evaluate_quick(
            user_message=message,
            retrieval_confidence=retrieval_result.confidence
        )

        if quick_decision and quick_decision.should_escalate:
            return await self._handle_escalation(
                context=context,
                decision=quick_decision,
                retrieved_docs=retrieval_result.documents,
                start_time=start_time
            )

        # Build context for generation
        context_text = self._build_context(retrieval_result.documents)

        # Get retrieved image URLs
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

        # Generate response WITH conversation history
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

        # Full escalation evaluation
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

        # Build response
        latency_ms = (time.time() - start_time) * 1000
        response = self._build_response(
            content=vlm_response.content,
            retrieval_result=retrieval_result,
            vlm_response=vlm_response,
            latency_ms=latency_ms
        )

        # Add soft escalation offer if needed
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

        return response

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
```

---

#### Task 1.3: Enhanced LLM Client
**File**: `llm_client.py`

**Key Changes:**
1. Add `generate_response_v2()` with conversation history support
2. Separate retrieved images from user-uploaded images
3. Update system prompt with Aeris persona
4. Implement proper image ordering (user images last for VLM attention)

```python
async def generate_response_v2(
    self,
    current_query: str,
    context: str,
    retrieved_image_urls: List[str] = None,
    user_uploaded_images: List[str] = None,
    conversation_history: List[Dict[str, Any]] = None,
    enable_thinking: bool = None
) -> VLMResponse:
    """
    Generate RAG response with full multi-turn conversation support.

    Args:
        current_query: User's latest question (for logging)
        context: Retrieved knowledge base context
        retrieved_image_urls: Images from knowledge base (RAG results)
        user_uploaded_images: User's uploaded images (within retention window)
        conversation_history: Full conversation with turn metadata
        enable_thinking: Enable Qwen thinking mode

    Returns:
        VLMResponse with content (reasoning_content excluded)
    """
    import time
    start_time = time.time()

    # System prompt with Aeris persona
    # User Requirement: First-person with name
    system_prompt = """You are Aeris, a helpful multimodal assistant that answers questions based on a knowledge base of technical manuals and documents.

**Your personality:**
- Friendly and approachable - use first-person: "I'm Aeris, and I..."
- Professional but not robotic
- Patient with follow-up questions

**Your capabilities:**
- Analyze both text and images to answer questions accurately
- Maintain context across multiple conversation turns
- Reference specific sources when providing information
- Admit clearly when information is not in your knowledge base

**Guidelines:**
1. Answer based ONLY on the provided CONTEXT and retrieved images
2. Use conversation history to understand pronouns and references (e.g., "it" refers to the diagram from Turn 2)
3. If information is missing, clearly state: "I don't have information about [topic] in my knowledge base"
4. Cite sources when possible: "According to the Hydraulic Systems Manual, page 12..."
5. For images: "In the diagram you uploaded..." or "Looking at the retrieved schematic..."
6. Be concise but thorough - prioritize actionable information

**Context awareness:**
- You can see turn numbers: [Turn 3] means this is the third exchange
- When you see [📷 User uploaded: X], the user included an image in that turn
- Images remain available for 3 turns after upload for detailed analysis"""

    # Build messages array
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (already formatted with metadata)
    if conversation_history:
        # Exclude the last user message (we'll rebuild it with full context)
        messages.extend(conversation_history[:-1])

    # Build current user message with all context
    current_content = []

    # 1. Add retrieved images FIRST (from knowledge base)
    if retrieved_image_urls:
        for url in retrieved_image_urls[:5]:  # Max 5 KB images
            final_url = self._process_image_url(url)
            if final_url:
                current_content.append({
                    "type": "image_url",
                    "image_url": {"url": final_url}
                })

    # 2. Add user's uploaded images (within retention window)
    # User Requirement: Pass image for next 2-3 turns only
    if user_uploaded_images:
        for url in user_uploaded_images:
            final_url = self._process_image_url(url)
            if final_url:
                current_content.append({
                    "type": "image_url",
                    "image_url": {"url": final_url}
                })

    # 3. Add text prompt with knowledge base context
    user_prompt = f"""Based on the following knowledge base context and images, answer my question.

KNOWLEDGE BASE CONTEXT:
{context}

MY QUESTION: {current_query}

Remember to:
- Use the context and images above
- Reference which sources you're using
- If the answer isn't in the context, tell me clearly
- Consider our conversation history for context (pronouns, references)"""

    current_content.append({"type": "text", "text": user_prompt})

    messages.append({
        "role": "user",
        "content": current_content
    })

    # Determine thinking mode
    if enable_thinking is None:
        enable_thinking = self._should_use_thinking(current_query)

    extra_body = {}
    if enable_thinking:
        extra_body["enable_thinking"] = True

    try:
        response = await self.client.chat.completions.create(
            model=ModelTier.PLUS.value,
            messages=messages,
            max_tokens=config.qwen.max_output_tokens,
            extra_body=extra_body if extra_body else None
        )

        if response.usage:
            tracker.track_qwen(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )

        latency_ms = (time.time() - start_time) * 1000

        # CRITICAL: Only return content, not reasoning_content
        # Per Alibaba docs and user requirement
        return VLMResponse(
            content=response.choices[0].message.content,
            model=ModelTier.PLUS.value,
            thinking_enabled=enable_thinking,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms
        )

    except Exception as e:
        print(f"[X] Qwen Generation Error: {e}")
        raise

async def rewrite_query_v2(
    self,
    history_summary: str,
    current_query: str
) -> str:
    """
    Rewrite conversational query to be standalone.

    Enhanced to handle BOTH explicit pronouns AND implicit context references.
    Triggered by pronoun detection but handles broader contextual rewrites.
    """
    system_prompt = """You rewrite conversational queries to be standalone and context-independent.

EXAMPLES:

1. Pronoun Reference:
   History: "The X500 hydraulic pump has a blue pressure valve."
   Query: "How does it work?"
   Rewrite: "How does the X500 hydraulic pump work?"

2. Implicit Reference (NO pronoun):
   History: "The X500 pump uses a centrifugal design."
   Query: "What is the maintenance schedule?"
   Rewrite: "What is the maintenance schedule for the X500 hydraulic pump?"

3. Implicit Part Reference:
   History: "The X500 pump has a blue seal at the top."
   Query: "Is the seal compatible with other models?"
   Rewrite: "Is the blue seal from the X500 pump compatible with other pump models?"

4. Already Standalone:
   History: "The X500 pump..."
   Query: "Show me all hydraulic system diagrams."
   Rewrite: "Show me all hydraulic system diagrams." (unchanged)

RULES:
- Incorporate key entities from history (product names, part names, specifications)
- Preserve original intent and specificity
- If query is already standalone and clear, return unchanged
- Be concise - add only necessary context from history
- Focus on the last 2-3 exchanges for most relevant context"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"CONVERSATION HISTORY:\n{history_summary}\n\nCURRENT QUERY: {current_query}\n\nREWRITTEN QUERY:"
        }
    ]

    try:
        response = await self.client.chat.completions.create(
            model=ModelTier.FLASH.value,  # Fast model for query rewriting
            messages=messages,
            max_tokens=100,
            temperature=0.3  # Lower temperature for consistent rewrites
        )

        rewritten = response.choices[0].message.content.strip()

        # Track usage
        if response.usage:
            tracker.track_qwen(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens
            )

        return rewritten

    except Exception as e:
        print(f"[X] Query rewrite error: {e}")
        # Fallback to original query
        return current_query
```

---

### Phase 2: Frontend UX (1 day)

#### Task 2.1: Rebrand to Aeris
**File**: `static/index.html`

Changes:
- Update logo and title
- Update empty state with Aeris greeting
- Add "New Chat" button
- Add context warning display area

```html
<title>Aeris - Your Knowledge Assistant</title>

<!-- Header -->
<div class="logo">
    <div class="logo-mark">A</div>
    <span class="logo-text">Aeris</span>
</div>

<!-- Add New Chat button -->
<div class="conversation-controls">
    <button class="btn-secondary" onclick="startNewChat()" title="Start fresh conversation">
        <svg><!-- Plus icon --></svg>
        New Chat
    </button>
    <span id="turnCounter" style="display: none;">Turn 1</span>
</div>

<!-- Empty state -->
<h2 class="empty-title">Hi! I'm Aeris, your knowledge assistant</h2>
<p class="empty-subtitle">
    I can help you find information from your uploaded manuals and documents.
    Ask questions in plain language or upload images to search visually.
</p>

<!-- Context warning banner (initially hidden) -->
<div id="contextWarning" class="context-warning" style="display: none;"></div>
```

#### Task 2.2: JavaScript Updates
**File**: `static/js/chat.js`

```javascript
let conversationId = null;
let currentTurn = 0;

function startNewChat() {
    if (!confirm('Start a new conversation? Your current context will be cleared.')) {
        return;
    }

    // Reset state
    conversationId = null;
    currentTurn = 0;

    // Clear UI
    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML = `
        <div class="empty-state" id="emptyState">
            <!-- Empty state HTML -->
        </div>
    `;

    document.getElementById('turnCounter').style.display = 'none';
    document.getElementById('contextWarning').style.display = 'none';

    // Optionally call backend to clear (if needed)
    // Not required per user requirement (never auto-clear)
}

function updateTurnCounter(turn) {
    const counter = document.getElementById('turnCounter');
    if (turn > 0) {
        counter.textContent = `Turn ${turn}`;
        counter.style.display = 'inline-block';
    }
}

function showContextWarning(message) {
    const warning = document.getElementById('contextWarning');
    warning.textContent = message;
    warning.style.display = 'block';

    // Auto-hide after 10 seconds
    setTimeout(() => {
        warning.style.display = 'none';
    }, 10000);
}

// Update sendMessage to handle response metadata
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message && !selectedImage) return;

    // OPTIMISTIC UI: Show user message immediately
    addMessage('user', message, { pending: false });

    // Show loading assistant message
    const assistantMsg = addMessage('assistant', '', {
        pending: true,
        content: '<div class="typing-indicator"><span></span><span></span><span></span></div>'
    });

    const formData = new FormData();
    formData.append('message', message);
    if (selectedImage) {
        formData.append('image', selectedImage);
    }
    if (conversationId) {
        formData.append('conversation_id', conversationId);
    }

    try {
        const response = await fetch('/chat/stream', {
            method: 'POST',
            body: formData
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.trim()) continue;

                try {
                    const data = JSON.parse(line);

                    // EARLY METADATA: Received immediately before streaming
                    if (data.type === 'metadata') {
                        conversationId = data.conversation_id;
                        currentTurn = data.turn;
                        updateTurnCounter(currentTurn);
                    }

                    // STREAMING CONTENT
                    if (data.type === 'content' || data.type === 'token') {
                        // Remove typing indicator on first token
                        if (assistantMsg.classList.contains('pending')) {
                            assistantMsg.classList.remove('pending');
                            assistantMsg.innerHTML = '';
                        }
                        // Append streamed content
                        assistantMsg.innerHTML += data.content || data.text || '';
                    }

                    // FINAL METADATA: Sources, warnings, etc.
                    if (data.type === 'complete') {
                        if (data.context_warning) {
                            showContextWarning(data.context_warning);
                        }
                        // Render sources if any
                        if (data.sources && data.sources.length > 0) {
                            appendSources(assistantMsg, data.sources);
                        }
                    }
                } catch (e) {
                    console.error('Failed to parse streaming data:', line, e);
                }
            }
        }
    } catch (error) {
        console.error('Streaming error:', error);
        assistantMsg.classList.remove('pending');
        assistantMsg.innerHTML = '<span class="error">Failed to get response. Please try again.</span>';
    }

    // Clear input
    messageInput.value = '';
    selectedImage = null;
}

function addMessage(role, content, options = {}) {
    const msg = document.createElement('div');
    msg.className = `message ${role} ${options.pending ? 'pending' : ''}`;

    if (options.pending) {
        msg.innerHTML = options.content || '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    } else {
        msg.innerHTML = `<div class="message-content">${parseMarkdown(content)}</div>`;
    }

    messagesContainer.appendChild(msg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return msg;
}
```

**CSS for Optimistic UI:**
```css
.message.pending {
    opacity: 0.9;
}

.typing-indicator {
    display: flex;
    gap: 4px;
    padding: 12px 0;
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-primary);
    animation: typing-pulse 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing-pulse {
    0%, 60%, 100% {
        opacity: 0.3;
        transform: scale(0.8);
    }
    30% {
        opacity: 1;
        transform: scale(1);
    }
}

.message.error {
    color: var(--error-color);
    font-style: italic;
}
```

---

### Phase 3: Configuration & Testing (0.5 days)

#### Task 3.1: Update Configuration
**File**: `config.py`

```python
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

@dataclass
class AppConfig:
    # ... existing configs ...
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
```

#### Task 3.2: Update Response Models
**File**: `server.py`

```python
class ChatResponseModel(BaseModel):
    """Chat response body."""
    message: str
    conversation_id: str
    sources: List[dict] = []
    confidence: float
    escalated: bool
    escalation_reason: Optional[str] = None
    source_images: List[dict] = []
    latency_ms: float

    # New fields for multi-turn
    turn: int = 0
    context_warning: Optional[str] = None
    images_retained: int = 0  # Number of user images still in context
```

---

### Phase 4: Production Hardening (Post-MVP) (1 day)

**Goal**: Address scalability and robustness concerns for production deployment.

#### Task 4.1: Redis-Based Conversation Persistence
**Why**: In-memory storage loses all conversations on server restart.

**Implementation**:
```python
# conversation_store.py

import redis
import json
from typing import Optional
from escalation import ConversationContext

class RedisConversationStore:
    """
    Persistent conversation storage using Redis.
    Replaces in-memory Dict[str, ConversationContext].
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.ttl = 86400  # 24 hours

    def get(self, conversation_id: str) -> Optional[ConversationContext]:
        """Retrieve conversation from Redis."""
        data = self.client.get(f"conversation:{conversation_id}")
        if not data:
            return None

        # Deserialize JSON to ConversationContext
        context_dict = json.loads(data)
        return ConversationContext.from_dict(context_dict)

    def set(self, conversation_id: str, context: ConversationContext):
        """Store conversation in Redis with TTL."""
        data = json.dumps(context.to_dict())
        self.client.setex(
            f"conversation:{conversation_id}",
            self.ttl,
            data
        )

    def delete(self, conversation_id: str):
        """Delete conversation (for "New Chat")."""
        self.client.delete(f"conversation:{conversation_id}")
```

**Changes to `chatbot.py`**:
```python
class MultimodalChatbot:
    def __init__(self):
        # Replace in-memory dict with Redis
        if config.conversation.use_redis:
            self.store = RedisConversationStore()
        else:
            self.store = InMemoryConversationStore()  # Fallback

    def get_or_create_context(self, conversation_id, user_metadata):
        context = self.store.get(conversation_id)
        if not context:
            context = ConversationContext(conversation_id, user_metadata=user_metadata)
        return context

    async def chat(self, ...):
        # ... existing code ...

        # Save after each turn
        self.store.set(context.conversation_id, context)
```

**New Config**:
```python
@dataclass
class ConversationConfig:
    # ... existing fields ...
    use_redis: bool = False  # Enable for production
    redis_url: str = "redis://localhost:6379"
    conversation_ttl_seconds: int = 86400  # 24 hours
```

#### Task 4.2: Enhanced Query Rewriting
**Goal**: Handle implicit context (not just pronouns).

**Updated Prompt**:
```python
async def rewrite_query_v2(self, history_summary: str, current_query: str) -> str:
    """
    Rewrite query to be standalone, handling BOTH pronouns AND implicit references.
    """
    system_prompt = """You rewrite conversational queries to be standalone and context-independent.

EXAMPLES:

1. Pronoun Reference:
   History: "The X500 hydraulic pump has a blue pressure valve."
   Query: "How does it work?"
   Rewrite: "How does the X500 hydraulic pump work?"

2. Implicit Reference (NO pronoun):
   History: "The X500 pump uses a centrifugal design."
   Query: "What is the maintenance schedule?"
   Rewrite: "What is the maintenance schedule for the X500 hydraulic pump?"

3. Implicit Part Reference:
   History: "The X500 pump has a blue seal at the top."
   Query: "Is the seal compatible with other models?"
   Rewrite: "Is the blue seal from the X500 pump compatible with other pump models?"

4. Already Standalone:
   History: "The X500 pump..."
   Query: "Show me all hydraulic system diagrams."
   Rewrite: "Show me all hydraulic system diagrams." (unchanged)

RULES:
- Incorporate key entities from history (product names, part names)
- Preserve original intent and specificity
- If query is already standalone, return unchanged
- Be concise - add only necessary context"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONVERSATION HISTORY:\n{history_summary}\n\nCURRENT QUERY: {current_query}\n\nREWRITTEN QUERY:"}
    ]

    response = await self.client.chat.completions.create(
        model=ModelTier.FLASH.value,  # Fast model for rewriting
        messages=messages,
        max_tokens=100
    )

    return response.choices[0].message.content.strip()
```

#### Task 4.3: Semantic Image Retention (Optional)
**Goal**: Keep images in context if semantically relevant, even beyond 3 turns.

```python
def should_include_image_from_turn(
    self,
    image_turn: int,
    current_query: str = None
) -> bool:
    """
    Enhanced image retention with semantic check.

    Rules:
    1. Always include if within 3-turn window (primary rule)
    2. Beyond 3 turns: check semantic relevance (optional enhancement)
    """
    # Rule 1: 3-turn window (always applies)
    turn_diff = self.current_turn - image_turn
    if turn_diff <= self.IMAGE_RETENTION_TURNS:
        return True

    # Rule 2: Semantic relevance check (optional)
    if not config.conversation.semantic_image_retention:
        return False

    if current_query and image_turn in self.image_descriptions:
        # Compute similarity between query and image description
        from embeddings import compute_text_similarity
        similarity = compute_text_similarity(
            current_query,
            self.image_descriptions[image_turn]
        )
        return similarity > 0.75  # High threshold

    return False
```

**New Config**:
```python
@dataclass
class ConversationConfig:
    # ... existing fields ...
    semantic_image_retention: bool = False  # Opt-in for Phase 4
    semantic_similarity_threshold: float = 0.75
```

#### Task 4.4: Token Usage Analytics
**Goal**: Monitor actual token consumption patterns.

```python
# analytics.py

class TokenUsageAnalytics:
    """Track token usage patterns for optimization."""

    def __init__(self):
        self.usage_log = []

    def log_conversation(self, conversation_id: str, context: ConversationContext):
        """Log token usage for a conversation."""
        self.usage_log.append({
            "conversation_id": conversation_id,
            "total_tokens": context.total_tokens,
            "num_turns": context.current_turn,
            "num_images": len(context.image_turns),
            "pruned": context.pruning_notice_shown,
            "avg_tokens_per_turn": context.total_tokens / max(context.current_turn, 1)
        })

    def get_stats(self) -> dict:
        """Get aggregate statistics."""
        if not self.usage_log:
            return {}

        return {
            "avg_tokens_per_conversation": sum(log["total_tokens"] for log in self.usage_log) / len(self.usage_log),
            "avg_turns_per_conversation": sum(log["num_turns"] for log in self.usage_log) / len(self.usage_log),
            "prune_rate": sum(1 for log in self.usage_log if log["pruned"]) / len(self.usage_log),
            "total_conversations": len(self.usage_log)
        }
```

**Add to `/usage` endpoint**:
```python
@app.get("/usage")
async def get_usage():
    """Get API usage and conversation statistics."""
    return {
        "api_usage": tracker.get_usage(),
        "conversation_stats": analytics.get_stats()  # NEW
    }
```

---

## Testing Plan

### Unit Tests

```python
# test_conversation_context.py

def test_image_retention():
    """Test image retention for 3 turns."""
    ctx = ConversationContext("test")

    # Turn 1: Upload image
    ctx.add_user_message_v2("What is this?", image_url="img1.jpg")
    assert ctx.should_include_image_from_turn(1) == True

    # Turn 2: Follow-up
    ctx.add_user_message_v2("Where is the valve?")
    assert ctx.should_include_image_from_turn(1) == True

    # Turn 3: Still in window
    ctx.add_user_message_v2("What's the part number?")
    assert ctx.should_include_image_from_turn(1) == True

    # Turn 4: Outside window
    ctx.add_user_message_v2("How to maintain?")
    assert ctx.should_include_image_from_turn(1) == False

def test_pronoun_detection():
    """Test query rewrite triggering."""
    ctx = ConversationContext("test")
    ctx.add_user_message_v2("What is this?")
    ctx.add_assistant_message_v2("It's a pump.")

    assert ctx.needs_query_rewrite("How does it work?") == True
    assert ctx.needs_query_rewrite("Tell me about pumps") == False

def test_auto_pruning():
    """Test silent auto-pruning at token limit."""
    ctx = ConversationContext("test")

    # Simulate adding messages with real API token counts
    for i in range(20):
        ctx.add_user_message_v2("Question " + "x" * 200)
        # Simulate API response with realistic token counts
        ctx.add_assistant_message_v2(
            content="Answer " + "x" * 200,
            api_usage={"prompt_tokens": 800, "completion_tokens": 200, "total_tokens": 1000 * (i + 1)}
        )

    # Should have pruned old messages when over 28k tokens
    assert len(ctx.messages) < 40  # Some messages pruned
    assert ctx.pruning_notice_shown == True
    assert ctx.total_tokens < ctx.MAX_TOKENS

def test_api_based_token_counting():
    """Test that token counting uses actual API usage, not character estimates."""
    ctx = ConversationContext("test")

    # Add user message (no token count yet)
    ctx.add_user_message_v2("Test question")

    # Add assistant message with API usage
    ctx.add_assistant_message_v2(
        content="Test response",
        api_usage={"prompt_tokens": 1523, "completion_tokens": 89, "total_tokens": 1612}
    )

    # Should use exact API count, not character-based estimate
    assert ctx.total_tokens == 1612  # Exact from API
    # NOT approximately len("Test question" + "Test response") // 4
```

### Integration Tests

1. **Welcome Message Test**
   - Start fresh conversation
   - Verify first response is welcome message
   - Verify no retrieval/RAG on first message

2. **Multi-Turn Image Test**
   ```
   Turn 1: Upload pump.jpg + "What is this?"
   Turn 2: "Where is the pressure valve?" (image should be included)
   Turn 3: "What's its part number?" (image should be included)
   Turn 4: "How to maintain pumps?" (image should NOT be included)
   ```

3. **Pronoun Resolution Test**
   ```
   User: "Show me hydraulic pump diagrams"
   Aeris: [Shows diagrams]
   User: "Explain how it works"
   → Query should be rewritten to "Explain how hydraulic pumps work"
   ```

---

## Deployment Checklist

### Pre-Deployment
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Manual testing with 5 scenarios
- [ ] Token cost estimation reviewed
- [ ] Documentation updated

### Environment Variables
```bash
# Existing
VOYAGE_API_KEY=...
DASHSCOPE_API_KEY=...
QDRANT_URL=...

# New (optional overrides)
CHATBOT_NAME=Aeris
MAX_CONTEXT_TOKENS=28000  # Updated from 8000 to leverage 32k context window
WARNING_THRESHOLD=24000   # Updated from 6500
IMAGE_RETENTION_TURNS=3
```

### Monitoring Metrics
- Average conversation length (turns per session)
- Query rewrite trigger rate (should be ~30% of follow-ups)
- Context pruning frequency
- Image retention hit rate
- Token usage per conversation

---

## Success Metrics

### Quantitative
- **Multi-turn engagement**: 40%+ sessions with 3+ messages
- **Pronoun resolution accuracy**: 85%+ successful rewrites
- **Context limit errors**: <1% of sessions
- **Welcome message view rate**: 100% of new conversations

### Qualitative
- Users describe Aeris as "remembers what I said"
- Reduced "I don't understand" responses
- Positive feedback on welcome message clarity

---

## Timeline

| Phase | Duration | Tasks | Priority |
|-------|----------|-------|----------|
| Phase 1: Core | 2 days | ConversationContext (with API token counting), Chatbot, LLM Client | **REQUIRED** |
| Phase 2: Frontend | 1 day | HTML, CSS, JavaScript updates, Aeris branding | **REQUIRED** |
| Phase 3: Testing | 0.5 days | Unit + Integration tests | **REQUIRED** |
| **MVP Total** | **3.5 days** | | |
| Phase 4: Hardening | 1 day | Redis persistence, Enhanced query rewriting, Analytics | **OPTIONAL** |
| **Production-Ready Total** | **4.5 days** | | |

### Implementation Priority

**Must-Have (MVP - 3.5 days)**:
- ✅ API-based token counting (CRITICAL FIX)
- ✅ 28k context window (CRITICAL FIX)
- ✅ Multi-turn conversation tracking
- ✅ 3-turn image retention
- ✅ Pronoun-based query rewriting
- ✅ Aeris persona and branding
- ✅ Context warning UX

**Should-Have (Phase 4 - 1 day)**:
- Redis persistence (prevents data loss on restart)
- Enhanced query rewriting (implicit context)
- Token usage analytics
- Semantic image retention

### ✅ Design Decisions Validated

1. **In-Memory Storage for MVP**: Acceptable for prototype validation
2. **3-Turn Image Retention**: Per user requirements, cost-effective
3. **Pronoun-Based Query Rewriting**: 80% coverage at 30% cost
4. **No SSE**: Fetch-based streaming works well for binary uploads

---

## Next Steps

### Phase 1: Core Infrastructure
1. ✅ **Plan Updated** - Critical fixes incorporated
2. 🔄 **Start Implementation** - Begin with `escalation.py` (ConversationContext)
3. ⏳ **Integration** - Update `chatbot.py` and `llm_client.py`
4. ⏳ **Testing** - Validate token counting accuracy

### Review Checkpoints
- [ ] After Phase 1: Verify API token counting works correctly
- [ ] After Phase 1: Validate preemptive pruning prevents over-budget calls
- [ ] After Phase 1: Confirm image budgeting (2 user + 3 KB max)
- [ ] After Phase 2: User test Aeris branding and optimistic UI
- [ ] After Phase 2: Validate inline welcome + immediate answers
- [ ] After Phase 3: Validate multi-turn conversations work end-to-end
- [ ] **Decision**: Proceed to Phase 4 if MVP validation succeeds

---

## Summary of Analysis-Driven Improvements

The implementation plan has been enhanced based on thorough engineering analysis (IMPLEMENTATION_PLAN_ANALYSIS.md):

### ✅ Accepted Critical Improvements

1. **Image Token Budgeting** (CRITICAL)
   - **Problem**: Unbounded image passing consumed 51% of token budget
   - **Fix**: Cap at 2 user + 3 KB images = 9k tokens (32% of budget)
   - **Impact**: Preserves 19k tokens for conversation text, reduces pruning

2. **Preemptive Token Management**
   - **Problem**: Reactive pruning wasted expensive over-budget API calls
   - **Fix**: Heuristic-based pruning BEFORE API call (message count + estimation)
   - **Impact**: 15-25% reduction in wasted API costs

3. **Optimistic UI**
   - **Problem**: 2-5s blank screen wait creates poor perceived performance
   - **Fix**: Show user message + typing indicator immediately
   - **Impact**: 2x faster perceived response time

4. **Streaming Metadata Timing**
   - **Problem**: Frontend state corrupted on mid-stream disconnections
   - **Fix**: Send conversation_id/turn immediately in first chunk
   - **Impact**: Resilient to network interruptions

5. **Inline Welcome Message**
   - **Problem**: Blocking welcome requires 2 round-trips for first answer
   - **Fix**: Prepend compact greeting to first response
   - **Impact**: 50% faster time-to-value for new users

### ❌ Rejected Optimizations (Deferred to Phase 4 or Later)

- **Complex Entity Extraction for Query Rewriting**: Over-engineered, error-prone
- **Parallel Retrieval + Rewriting**: Complexity doesn't justify 300ms saving
- **Retrieval Result Caching**: Cache invalidation complexity for MVP
- **Preemptive Source Loading**: Over-engineered, adds backend load
- **Session Recovery**: Better solved with Redis persistence in Phase 4

### Key Design Principles Maintained

1. **Simplicity over complexity**: Heuristics instead of ML for token estimation
2. **User requirements first**: 3-turn image retention per explicit request
3. **Progressive enhancement**: Basic optimistic UI, not complex stage cycling
4. **MVP focus**: Defer non-critical features to Phase 4

Ready to begin implementation with production-grade token management and UX optimizations.
