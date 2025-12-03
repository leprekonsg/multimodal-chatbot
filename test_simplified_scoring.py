#!/usr/bin/env python3
"""
Test simplified component scoring approach.
Validates the cascading logic without weighted combination.
"""
from component_scorer_simple import ComponentScorer, QueryIntent


def test_exact_cases():
    """Test exact match scenarios."""
    scorer = ComponentScorer()

    print("=" * 80)
    print("TEST 1: Exact Matches")
    print("=" * 80)

    cases = [
        ("tiedown ring", "tiedown", 1.0, "Exact substring"),
        ("reset button", "reset", 1.0, "Exact token match"),
        ("emergency shutoff valve", "shutoff", 1.0, "Exact substring in compound"),
    ]

    for label, query, expected_min, description in cases:
        score = scorer.score_component(label, query)
        status = "✓" if score >= expected_min else "✗"
        print(f"{status} {description}")
        print(f"  Label: '{label}'")
        print(f"  Query: '{query}'")
        print(f"  Score: {score:.3f} (expected >= {expected_min})")
        print()


def test_compound_words():
    """Test compound word handling."""
    scorer = ComponentScorer()

    print("=" * 80)
    print("TEST 2: Compound Word Matching")
    print("=" * 80)

    cases = [
        ("aft fuselage tiedown ring", "tie down", 0.5, "Compound: 'tie down' → 'tiedown'"),
        ("emergency shut down procedure", "shutdown", 0.5, "Compound: 'shutdown' → 'shut down'"),
        ("power button control", "pwr button", 0.3, "Abbreviation + word"),
    ]

    for label, query, expected_min, description in cases:
        score = scorer.score_component(label, query)
        status = "✓" if score >= expected_min else "✗"
        print(f"{status} {description}")
        print(f"  Label: '{label}'")
        print(f"  Query: '{query}'")
        print(f"  Score: {score:.3f} (expected >= {expected_min})")
        print()


def test_false_positives():
    """Test that unrelated labels score low."""
    scorer = ComponentScorer()

    print("=" * 80)
    print("TEST 3: False Positive Prevention")
    print("=" * 80)

    cases = [
        ("diagram caption text", "reset button", 0.3, "Should be low - unrelated"),
        ("Square Knot tying diagram", "tiedown procedures", 0.3, "Tangentially related"),
        ("specification document header", "emergency valve", 0.3, "Unrelated"),
    ]

    for label, query, expected_max, description in cases:
        score = scorer.score_component(label, query)
        status = "✓" if score <= expected_max else "✗"
        print(f"{status} {description}")
        print(f"  Label: '{label}'")
        print(f"  Query: '{query}'")
        print(f"  Score: {score:.3f} (expected <= {expected_max})")
        print()


def test_actual_user_query():
    """Test with the actual problematic query from logs."""
    scorer = ComponentScorer()

    print("=" * 80)
    print("TEST 4: Real User Query - 'Tie down procedures for helicopters'")
    print("=" * 80)

    query = "Tie down procedures for helicopters"

    components = [
        {"label": "diagram of aft fuselage tiedown ring"},
        {"label": "diagram of nose landing gear tiedown loop"},
        {"label": "diagram of underside of wing tiedown loop"},
        {"label": "diagram of main gear wheel tiedown loop"},
        {"label": "diagram showing Bowline Knot tying steps"},
        {"label": "diagram showing Square Knot tying technique with arrows"},
        {"label": "diagram illustrating knot-tying technique with arrows"},
        {"label": "text label indicating 'Over' and 'Under' in knot diagram"},
        {"label": "caption for Bowline Knot diagram"},
        {"label": "caption for Square Knot diagram"},
        {"label": "close-up diagram of Square Knot with 'Over' and 'Under' labels"},
    ]

    scored = scorer.score_components(components.copy(), query, QueryIntent.TEXTUAL_SEARCH)

    # Sort by score
    scored.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    print(f"\nQuery: '{query}'\n")
    print("Ranking:")
    print("-" * 80)

    high_count = 0
    medium_count = 0
    low_count = 0

    for i, comp in enumerate(scored, 1):
        score = comp.get('relevance_score', 0)
        label = comp['label']

        if score >= 0.5:
            tier = "🟢 HIGH"
            high_count += 1
        elif score >= 0.25:
            tier = "🟡 MEDIUM"
            medium_count += 1
        else:
            tier = "⚪ LOW"
            low_count += 1

        print(f"{i:2}. [{tier}] {score:.3f} | {label[:65]}")

    print("\n" + "=" * 80)
    print(f"Distribution: {high_count} HIGH, {medium_count} MEDIUM, {low_count} LOW")
    print("=" * 80)
    print("\nExpected: 4 tiedown components HIGH (containing 'tiedown')")
    print("         4-5 knot components MEDIUM-LOW (related but not primary match)")
    print("         2-3 text/caption components LOW (tangential)")


def test_intent_aware_thresholds():
    """Test that intent changes visibility threshold."""
    scorer = ComponentScorer()

    print("\n" + "=" * 80)
    print("TEST 5: Intent-Aware Thresholds")
    print("=" * 80)

    label = "emergency shutdown button"
    query = "shutoff"

    score = scorer.score_component(label, query)

    print(f"\nLabel: '{label}'")
    print(f"Query: '{query}'")
    print(f"Score: {score:.3f}\n")

    for intent in [QueryIntent.VISUAL_SEARCH, QueryIntent.TEXTUAL_SEARCH, QueryIntent.EXACT_MATCH]:
        threshold = scorer.INTENT_THRESHOLDS[intent]
        visible = score >= threshold
        status = "✓ VISIBLE" if visible else "✗ HIDDEN"

        print(f"{intent.value:20} | Threshold: {threshold:.2f} | {status}")

    print("\nExpected: Visible for VISUAL/TEXTUAL, hidden for EXACT_MATCH")


if __name__ == "__main__":
    test_exact_cases()
    test_compound_words()
    test_false_positives()
    test_actual_user_query()
    test_intent_aware_thresholds()

    print("\n" + "=" * 80)
    print("All tests complete!")
    print("=" * 80)
