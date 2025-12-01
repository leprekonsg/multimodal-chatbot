"""
Unit tests for ConversationContext multi-turn conversation support.

Tests cover:
- Image retention logic (3-turn window)
- Token counting with actual API usage
- Query rewrite triggering (pronoun detection)
- Auto-pruning at token limits
- Preemptive pruning before API calls
"""

import pytest
from escalation import ConversationContext


class TestImageRetention:
    """Test image retention for 3-turn window."""

    def test_image_included_in_turn_1(self):
        """Image should be included in the turn it's uploaded."""
        ctx = ConversationContext("test_conv_1")
        ctx.add_user_message_v2("What is this?", image_url="http://localhost/img1.jpg", image_filename="pump.jpg")

        assert ctx.should_include_image_from_turn(1) == True
        assert 1 in ctx.image_turns

    def test_image_included_turn_2_and_3(self):
        """Image should be included in turns 2 and 3 after upload."""
        ctx = ConversationContext("test_conv_2")

        # Turn 1: Upload image
        ctx.add_user_message_v2("What is this?", image_url="http://localhost/img1.jpg")
        assert ctx.current_turn == 1
        assert ctx.should_include_image_from_turn(1) == True

        # Turn 2: Follow-up
        ctx.add_user_message_v2("Where is the valve?")
        assert ctx.current_turn == 2
        assert ctx.should_include_image_from_turn(1) == True

        # Turn 3: Still in window
        ctx.add_user_message_v2("What's the part number?")
        assert ctx.current_turn == 3
        assert ctx.should_include_image_from_turn(1) == True

    def test_image_excluded_turn_4(self):
        """Image should be excluded from turn 4 onwards."""
        ctx = ConversationContext("test_conv_3")

        # Turn 1: Upload image
        ctx.add_user_message_v2("What is this?", image_url="http://localhost/img1.jpg")

        # Turn 2, 3, 4: Follow-ups
        ctx.add_user_message_v2("Follow-up 1")
        ctx.add_user_message_v2("Follow-up 2")
        ctx.add_user_message_v2("How to maintain?")  # Turn 4

        assert ctx.current_turn == 4
        assert ctx.should_include_image_from_turn(1) == False

    def test_multiple_images_within_retention(self):
        """Multiple images within retention window should all be available."""
        ctx = ConversationContext("test_conv_4")

        # Turn 1: Upload first image
        ctx.add_user_message_v2("Image 1", image_url="http://localhost/img1.jpg")

        # Turn 2: Ask about first
        ctx.add_user_message_v2("Details about it?")

        # Turn 3: Upload second image
        ctx.add_user_message_v2("Image 2", image_url="http://localhost/img2.jpg")

        # Turn 4: Both should be available
        ctx.add_user_message_v2("Compare them")

        assert ctx.should_include_image_from_turn(1) == True  # 4 - 1 = 3 (boundary)
        assert ctx.should_include_image_from_turn(3) == True  # 4 - 3 = 1

    def test_get_active_images_respects_limit(self):
        """get_active_images should limit to max_images parameter."""
        ctx = ConversationContext("test_conv_5")

        # Upload 3 images
        ctx.add_user_message_v2("Image 1", image_url="http://localhost/img1.jpg")
        ctx.add_user_message_v2("Image 2", image_url="http://localhost/img2.jpg")
        ctx.add_user_message_v2("Image 3", image_url="http://localhost/img3.jpg")

        # Get active images with limit of 2
        active = ctx.get_active_images(max_images=2)

        assert len(active) <= 2
        assert "http://localhost/img3.jpg" in active  # Most recent first


class TestTokenCounting:
    """Test API-based token counting (not character-based estimates)."""

    def test_token_counting_uses_api_usage(self):
        """Token count should come from API response, not character estimates."""
        ctx = ConversationContext("test_token_1")

        # Add user message (no token count yet)
        ctx.add_user_message_v2("Test question about pumps")
        assert ctx.total_tokens == 0  # Not counted until API response

        # Add assistant message with actual API usage
        ctx.add_assistant_message_v2(
            content="Test response about pumps",
            api_usage={
                "prompt_tokens": 1523,
                "completion_tokens": 89,
                "total_tokens": 1612
            }
        )

        # Should use exact API count, not character-based estimate
        assert ctx.total_tokens == 1612

    def test_token_count_accumulates(self):
        """Token count should accumulate across turns."""
        ctx = ConversationContext("test_token_2")

        # Turn 1
        ctx.add_user_message_v2("First question")
        ctx.add_assistant_message_v2("First response", api_usage={"total_tokens": 1000})
        assert ctx.total_tokens == 1000

        # Turn 2
        ctx.add_user_message_v2("Second question")
        ctx.add_assistant_message_v2("Second response", api_usage={"total_tokens": 2000})
        # API usage is total, not incremental
        assert ctx.total_tokens == 2000

    def test_token_counting_with_images(self):
        """Token counting should work with multimodal content."""
        ctx = ConversationContext("test_token_3")

        # User uploads image and asks question
        ctx.add_user_message_v2(
            "What does this show?",
            image_url="http://localhost/img.jpg",
            image_filename="diagram.jpg"
        )
        assert ctx.total_tokens == 0

        # Response includes image processing tokens
        ctx.add_assistant_message_v2(
            content="This shows a hydraulic system diagram.",
            api_usage={
                "prompt_tokens": 2500,  # Higher due to image processing
                "completion_tokens": 50,
                "total_tokens": 2550
            }
        )
        assert ctx.total_tokens == 2550


class TestPronounDetection:
    """Test query rewrite triggering based on pronoun detection."""

    def test_needs_rewrite_with_pronouns(self):
        """Query with pronouns should trigger rewrite."""
        ctx = ConversationContext("test_pronouns_1")

        # First message - no history, should not rewrite
        assert ctx.needs_query_rewrite("How does it work?") == False

        # Add history
        ctx.add_user_message_v2("What is a hydraulic pump?")
        ctx.add_assistant_message_v2("A hydraulic pump is...", api_usage={"total_tokens": 500})

        # Now pronouns should trigger rewrite
        assert ctx.needs_query_rewrite("How does it work?") == True
        assert ctx.needs_query_rewrite("Where is that located?") == True
        assert ctx.needs_query_rewrite("What about them?") == True

    def test_no_rewrite_without_pronouns(self):
        """Query without pronouns should not trigger rewrite."""
        ctx = ConversationContext("test_pronouns_2")

        ctx.add_user_message_v2("Tell me about pumps")
        ctx.add_assistant_message_v2("Pumps are...", api_usage={"total_tokens": 500})

        # No pronouns - should not rewrite
        assert ctx.needs_query_rewrite("Show me all diagrams") == False
        assert ctx.needs_query_rewrite("What is maintenance?") == False

    def test_pronoun_variations(self):
        """Test various pronoun patterns."""
        ctx = ConversationContext("test_pronouns_3")
        ctx.add_user_message_v2("History")
        ctx.add_assistant_message_v2("Content", api_usage={"total_tokens": 500})

        pronouns_to_test = [
            ("How does it work?", True),
            ("Where is that?", True),
            ("What about them?", True),
            ("These are important", True),
            ("Its properties are...", True),
            ("Their specifications", True),
            ("Show diagrams", False),
            ("Explain pumps", False),
        ]

        for query, should_rewrite in pronouns_to_test:
            result = ctx.needs_query_rewrite(query)
            assert result == should_rewrite, f"Query '{query}' should rewrite={should_rewrite}"


class TestAutoPruning:
    """Test auto-pruning when token limit is exceeded."""

    def test_auto_prune_when_over_limit(self):
        """Messages should be pruned when total tokens exceed MAX_TOKENS."""
        ctx = ConversationContext("test_prune_1")

        # Add messages that will exceed MAX_TOKENS
        for i in range(5):
            ctx.add_user_message_v2(f"Question {i}")
            ctx.add_assistant_message_v2(
                f"Answer {i}",
                api_usage={
                    "prompt_tokens": 5000 + (i * 1000),
                    "completion_tokens": 2000,
                    "total_tokens": (i + 1) * 8000  # Progressively higher
                }
            )

        # Last total would be 40k, exceeding MAX_TOKENS (28k)
        # Auto-prune should have triggered
        assert ctx.total_tokens <= ctx.MAX_TOKENS
        assert len(ctx.messages) < 10  # Some messages should be pruned

    def test_prune_notice_shown(self):
        """Pruning notice should be set when auto-pruning occurs."""
        ctx = ConversationContext("test_prune_2")
        assert ctx.pruning_notice_shown == False

        # Add enough messages to trigger pruning
        for i in range(6):
            ctx.add_user_message_v2(f"Q{i}")
            ctx.add_assistant_message_v2(
                f"A{i}",
                api_usage={"total_tokens": (i + 1) * 6000}
            )

        # Should show pruning notice
        assert ctx.pruning_notice_shown == True

    def test_minimum_messages_kept(self):
        """Auto-pruning should keep at least 4 messages."""
        ctx = ConversationContext("test_prune_3")

        # Add many messages with high token counts
        for i in range(10):
            ctx.add_user_message_v2(f"Q{i}")
            ctx.add_assistant_message_v2(
                f"A{i}",
                api_usage={"total_tokens": 30000}  # Always over limit
            )

        # Should still have some recent messages
        assert len(ctx.messages) >= 4


class TestPreemptivePruning:
    """Test preemptive pruning to prevent over-budget API calls."""

    def test_prune_with_too_many_messages(self):
        """Should prune preemptively if > 40 messages."""
        ctx = ConversationContext("test_preempt_1")

        # Add 21 user-assistant pairs (42 messages)
        for i in range(21):
            ctx.add_user_message_v2(f"Question {i}")
            ctx.add_assistant_message_v2(f"Answer {i}", api_usage={"total_tokens": 1000 * (i + 1)})

        assert len(ctx.messages) > 40
        assert ctx.should_prune_preemptively() == True

    def test_prune_with_high_estimated_load(self):
        """Should prune if estimated tokens approach 85% of max."""
        ctx = ConversationContext("test_preempt_2")

        # Build up token count to 23,800 (85% of 28k)
        ctx.total_tokens = 23800
        num_active_images = 5

        # Estimated: 23800 + (5 * 1800) + 500 = 32,300 > 28,000 * 0.85
        assert ctx.should_prune_preemptively(num_active_images=num_active_images) == True

    def test_no_prune_when_under_threshold(self):
        """Should not prune if well under limits."""
        ctx = ConversationContext("test_preempt_3")

        # Add just a few messages
        ctx.add_user_message_v2("Q1")
        ctx.add_assistant_message_v2("A1", api_usage={"total_tokens": 500})

        assert ctx.should_prune_preemptively(num_active_images=1) == False


class TestWarningMessages:
    """Test context limit warning messages."""

    def test_warning_when_approaching_limit(self):
        """Should return warning when approaching WARNING_THRESHOLD."""
        ctx = ConversationContext("test_warn_1")

        # Set tokens just under warning threshold
        ctx.total_tokens = 24500

        warning = ctx.get_warning_message()
        assert warning is not None
        assert "context limit" in warning.lower()

    def test_no_warning_when_under_threshold(self):
        """Should not warn when well under threshold."""
        ctx = ConversationContext("test_warn_2")
        ctx.total_tokens = 20000

        assert ctx.get_warning_message() is None

    def test_pruning_notice_message(self):
        """Should show notice after pruning."""
        ctx = ConversationContext("test_warn_3")
        ctx.pruning_notice_shown = True

        warning = ctx.get_warning_message()
        assert warning is not None
        assert "earlier messages" in warning.lower()
        assert ctx.pruning_notice_shown == False  # Should reset


class TestFormattedMessagesForLLM:
    """Test message formatting for LLM with metadata."""

    def test_format_with_turn_numbers(self):
        """Formatted messages should include turn numbers."""
        ctx = ConversationContext("test_format_1")

        ctx.add_user_message_v2("Question 1")
        ctx.add_assistant_message_v2("Answer 1", api_usage={"total_tokens": 500})

        formatted = ctx.get_messages_for_llm_v2()

        # Should have 2 messages
        assert len(formatted) == 2
        # First message should have turn metadata
        assert "[Turn 1]" in str(formatted[0])

    def test_format_with_image_markers(self):
        """Formatted messages should include image upload markers."""
        ctx = ConversationContext("test_format_2")

        ctx.add_user_message_v2(
            "What is this?",
            image_url="http://localhost/img.jpg",
            image_filename="diagram.jpg"
        )
        ctx.add_assistant_message_v2("It's a diagram", api_usage={"total_tokens": 500})

        formatted = ctx.get_messages_for_llm_v2()

        # Should have image marker
        assert "[📷 User uploaded: diagram.jpg]" in str(formatted[0])

    def test_assistant_messages_passthrough(self):
        """Assistant messages should pass through without modification."""
        ctx = ConversationContext("test_format_3")

        ctx.add_user_message_v2("Question")
        assistant_content = "This is the assistant's response"
        ctx.add_assistant_message_v2(assistant_content, api_usage={"total_tokens": 500})

        formatted = ctx.get_messages_for_llm_v2()

        # Assistant message should be unchanged
        assert formatted[1]["content"] == assistant_content


class TestIntegration:
    """Integration tests for complete conversation flows."""

    def test_full_conversation_with_image(self):
        """Test complete multi-turn conversation with image."""
        ctx = ConversationContext("test_integration_1")

        # Turn 1: User uploads image
        ctx.add_user_message_v2(
            "What does this pump diagram show?",
            image_url="http://localhost/pump.jpg",
            image_filename="pump_diagram.jpg"
        )
        assert ctx.current_turn == 1

        # Assistant responds
        ctx.add_assistant_message_v2(
            "This is a centrifugal pump diagram showing...",
            api_usage={"prompt_tokens": 2000, "completion_tokens": 100, "total_tokens": 2100}
        )
        assert ctx.total_tokens == 2100

        # Turn 2: Follow-up about the diagram
        ctx.add_user_message_v2("Where is the impeller in it?")
        assert ctx.current_turn == 2
        assert ctx.should_include_image_from_turn(1) == True

        # Turn 3: Another follow-up
        ctx.add_user_message_v2("What's the pressure rating?")
        assert ctx.current_turn == 3
        assert ctx.should_include_image_from_turn(1) == True

        # Turn 4: Image should no longer be included
        ctx.add_user_message_v2("How do I maintain it?")
        assert ctx.current_turn == 4
        assert ctx.should_include_image_from_turn(1) == False

    def test_conversation_with_pronoun_rewrite(self):
        """Test conversation requiring pronoun-based query rewrite."""
        ctx = ConversationContext("test_integration_2")

        # Turn 1
        ctx.add_user_message_v2("Tell me about X500 hydraulic pumps")
        ctx.add_assistant_message_v2("The X500 is a centrifugal pump...", api_usage={"total_tokens": 800})

        # Turn 2: Should trigger rewrite due to "it"
        assert ctx.needs_query_rewrite("How does it work?") == True

        ctx.add_user_message_v2("How does it work?")
        assert ctx.current_turn == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
