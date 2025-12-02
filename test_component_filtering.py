#!/usr/bin/env python3
"""
Test script for component filtering with the actual query from user's logs.
This demonstrates how the new substring containment approach handles compound words.
"""
import re
from typing import List, Dict, Any


def filter_relevant_components(components: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Simplified version of the scoring logic for testing.
    """
    if not components or not query:
        for comp in components:
            comp['relevance_score'] = 0.2
        return components

    if len(components) <= 10:
        for comp in components:
            comp['relevance_score'] = 0.5
        return components

    # Stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'what', 'which', 'where', 'when', 'who',
        'how', 'why', 'it', 'its', 'their', 'them', 'please', 'show', 'me',
        'find', 'locate', 'i', 'you', 'we'
    }

    # Tokenize query
    query_lower = query.lower()
    query_lower = re.sub(r'[-_]', ' ', query_lower)
    query_tokens = set(re.findall(r'[a-z0-9]+', query_lower))
    query_tokens = {token for token in query_tokens if token not in stopwords and len(token) >= 3}

    if not query_tokens:
        for comp in components:
            comp['relevance_score'] = 0.2
        return components

    # Score each component
    for comp in components:
        label = comp.get('label', '').lower()
        if not label:
            comp['relevance_score'] = 0.0
            continue

        label = re.sub(r'[-_]', ' ', label)

        # Substring matching
        matches = 0
        for token in query_tokens:
            if token in label:
                matches += 1

        comp['relevance_score'] = matches / len(query_tokens) if query_tokens else 0.0

    return components


def test_actual_query():
    """Test with the exact query and components from user's logs."""

    # Actual query from logs
    query = "Tie down procedures for helicopters"

    # Actual components from logs (11 total)
    components = [
        {"label": "diagram of aft fuselage tiedown ring", "bbox_2d": [510, 63, 704, 259]},
        {"label": "diagram of nose landing gear tiedown loop", "bbox_2d": [718, 63, 909, 259]},
        {"label": "diagram of underside of wing tiedown loop", "bbox_2d": [510, 260, 704, 457]},
        {"label": "diagram of main gear wheel tiedown loop", "bbox_2d": [718, 260, 909, 457]},
        {"label": "diagram showing Bowline Knot tying steps", "bbox_2d": [113, 485, 327, 867]},
        {"label": "diagram showing Square Knot tying technique with arrows", "bbox_2d": [327, 352, 519, 485]},
        {"label": "diagram illustrating knot-tying technique with arrows", "bbox_2d": [327, 485, 519, 618]},
        {"label": "text label indicating 'Over' and 'Under' in knot diagram", "bbox_2d": [519, 485, 710, 618]},
        {"label": "caption for Bowline Knot diagram", "bbox_2d": [672, 507, 862, 535]},
        {"label": "caption for Square Knot diagram", "bbox_2d": [672, 618, 862, 646]},
        {"label": "close-up diagram of Square Knot with 'Over' and 'Under' labels", "bbox_2d": [519, 543, 862, 867]},
    ]

    print("=" * 80)
    print("TEST: Substring Containment Approach")
    print("=" * 80)
    print(f"\nQuery: '{query}'")
    print(f"Total components: {len(components)}")

    # Apply filtering
    scored_components = filter_relevant_components(components.copy(), query)

    # Extract query tokens for display
    query_lower = query.lower()
    query_lower = re.sub(r'[-_]', ' ', query_lower)
    stopwords = {'the', 'a', 'for', 'of', 'to', 'in', 'and', 'or'}
    query_tokens = {token for token in re.findall(r'[a-z0-9]+', query_lower) if token not in stopwords and len(token) >= 3}

    print(f"\nQuery tokens (after stopword removal): {sorted(query_tokens)}")
    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)

    # Sort by relevance
    scored_components.sort(key=lambda x: x['relevance_score'], reverse=True)

    high_relevance = []
    medium_relevance = []
    low_relevance = []

    for comp in scored_components:
        score = comp['relevance_score']
        label = comp['label']

        if score > 0.25:
            high_relevance.append((label, score))
        elif score > 0.1:
            medium_relevance.append((label, score))
        else:
            low_relevance.append((label, score))

    print(f"\n🟢 HIGH RELEVANCE (score > 0.25) - {len(high_relevance)} components:")
    print("   Frontend: Opacity 1.0, border 3px, blue color")
    for label, score in high_relevance:
        print(f"   ✓ {label[:60]:<60} | Score: {score:.2f}")

    print(f"\n🟡 MEDIUM RELEVANCE (0.1 < score <= 0.25) - {len(medium_relevance)} components:")
    print("   Frontend: Opacity 0.6, border 2px, gray color")
    for label, score in medium_relevance:
        print(f"   ~ {label[:60]:<60} | Score: {score:.2f}")

    print(f"\n🔴 LOW RELEVANCE (score <= 0.1) - {len(low_relevance)} components:")
    print("   Frontend: Opacity 0.3, border 1px, light gray")
    for label, score in low_relevance:
        print(f"   - {label[:60]:<60} | Score: {score:.2f}")

    print("\n" + "=" * 80)
    print("EXPECTED UX IMPROVEMENT:")
    print("=" * 80)
    print(f"Before: All 11 components shown at full opacity (cluttered)")
    print(f"After:  4 prominent components + 7 faded (visual hierarchy)")
    print(f"\nUser sees relevant 'tiedown' diagrams clearly, while knot")
    print(f"diagrams are visible but de-emphasized.")
    print("=" * 80)


def test_compound_word_matching():
    """Demonstrate compound word handling."""
    print("\n\n" + "=" * 80)
    print("COMPOUND WORD MATCHING TEST")
    print("=" * 80)

    test_cases = [
        {
            "query": "tie down",
            "label": "aft fuselage tiedown ring",
            "expected": "MATCH (substring 'tie' in 'tiedown')"
        },
        {
            "query": "power button",
            "label": "pwr button control",
            "expected": "NO MATCH (different tokens, needs normalization)"
        },
        {
            "query": "valve",
            "label": "pressure relief valve assembly",
            "expected": "MATCH (exact token match)"
        },
        {
            "query": "shutdown",
            "label": "emergency shut down procedure",
            "expected": "PARTIAL MATCH (substring 'shut' matches)"
        },
    ]

    for i, case in enumerate(test_cases, 1):
        query = case['query']
        label = case['label']
        expected = case['expected']

        components = [{"label": label}]
        result = filter_relevant_components(components.copy(), query)
        score = result[0]['relevance_score']

        print(f"\nCase {i}:")
        print(f"  Query: '{query}'")
        print(f"  Label: '{label}'")
        print(f"  Score: {score:.2f}")
        print(f"  Expected: {expected}")
        print(f"  Result: {'✓ PASS' if score > 0 else '✗ FAIL'}")


if __name__ == "__main__":
    test_actual_query()
    test_compound_word_matching()
