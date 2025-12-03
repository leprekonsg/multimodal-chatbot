"""
Simplified Component Scoring - Cascading Logic (No Weighted Combination)

Philosophy: Priority-based scoring, not weighted averaging.
- Try exact match first → 1.0
- Fall back to n-gram similarity → 0.5-1.0
- Last resort: token overlap → 0.0-0.5

No configuration surface, no tuning, no fuzzy matching for imaginary typos.
"""
import re
from typing import List, Dict, Any, Set, Optional
from enum import Enum


class QueryIntent(Enum):
    """Query intent types for adaptive thresholds."""
    VISUAL_SEARCH = "visual"
    TEXTUAL_SEARCH = "textual"
    MULTIMODAL = "multimodal"
    EXACT_MATCH = "exact"
    SIMILARITY = "similarity"
    GENERAL = "general"


class ComponentScorer:
    """
    Simple component scorer with cascading priority logic.
    No weighted combination, no configuration hell.
    """

    # Stopwords to filter from tokens
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'what', 'which', 'where', 'when', 'who',
        'how', 'why', 'it', 'its', 'their', 'them', 'please', 'show', 'me',
        'find', 'locate', 'i', 'you', 'we', 'they'
    }

    # Intent-specific thresholds (what score is "visible")
    INTENT_THRESHOLDS = {
        QueryIntent.VISUAL_SEARCH: 0.15,    # Exploratory - show more
        QueryIntent.EXACT_MATCH: 0.60,      # Precise - show fewer
        QueryIntent.TEXTUAL_SEARCH: 0.25,   # Balanced
        QueryIntent.MULTIMODAL: 0.25,
        QueryIntent.SIMILARITY: 0.15,
        QueryIntent.GENERAL: 0.25,
    }

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text for matching: lowercase, remove hyphens/underscores/spaces.
        Handles compound words: "tie-down" → "tiedown"
        """
        if not text:
            return ""
        return re.sub(r'[-_\s]+', '', text.lower())

    @staticmethod
    def char_ngrams(text: str, n: int = 3) -> Set[str]:
        """
        Generate character n-grams for fuzzy matching.
        Example: "tiedown" with n=3 → {'tie', 'ied', 'edo', 'dow', 'own'}
        """
        if not text or len(text) < n:
            return set()
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    @staticmethod
    def tokenize(text: str) -> Set[str]:
        """Extract meaningful tokens (no stopwords, length >= 3)."""
        if not text:
            return set()

        # Lowercase and split on non-alphanumeric
        tokens = set(re.findall(r'[a-z0-9]+', text.lower()))

        # Filter stopwords and short tokens
        tokens = {t for t in tokens if t not in ComponentScorer.STOPWORDS and len(t) >= 3}

        return tokens

    def score_component(
        self,
        label: str,
        query: str
    ) -> float:
        """
        Score a single component using cascading priority logic.

        Priority order:
        1. Exact token match → 1.0
        2. N-gram similarity per token (handles compounds) → 0.6-0.95
        3. Token overlap (fallback) → 0.0-0.5

        Fixed score ranges (no overlap):
        - Exact: 1.0
        - N-gram high: 0.7-0.95
        - N-gram low: 0.6-0.7
        - Token overlap: 0.0-0.5

        Returns:
            Score in range [0.0, 1.0]
        """
        if not label or not query:
            return 0.2  # Default low score

        # Normalize label once
        label_norm = self.normalize(label)
        label_tokens = self.tokenize(label)

        # Extract query tokens (avoid dilution from multi-word queries)
        query_tokens = self.tokenize(query)

        if not query_tokens:
            return 0.2

        # Strategy 1: Exact token match (highest confidence)
        # Check if any query token is in label tokens OR in normalized label
        for q_token in query_tokens:
            if q_token in label_tokens or q_token in label_norm:
                # Check how many tokens match
                exact_matches = sum(1 for qt in query_tokens if qt in label_tokens or qt in label_norm)
                if exact_matches == len(query_tokens):
                    return 1.0  # All tokens found
                elif exact_matches >= max(1, len(query_tokens) // 2):
                    return 0.95  # Most tokens found

        # Strategy 2: N-gram similarity per token (handles compound words)
        # Check each query token against normalized label
        best_ngram_score = 0.0
        label_ngrams = self.char_ngrams(label_norm, n=3)

        for q_token in query_tokens:
            # Normalize individual token
            q_token_norm = self.normalize(q_token)
            q_ngrams = self.char_ngrams(q_token_norm, n=3)

            if q_ngrams and label_ngrams:
                intersection = len(q_ngrams & label_ngrams)
                union = len(q_ngrams | label_ngrams)
                jaccard = intersection / union if union > 0 else 0.0

                if jaccard > best_ngram_score:
                    best_ngram_score = jaccard

        # Map jaccard similarity to 0.6-0.95 range (below exact, above token overlap)
        if best_ngram_score > 0.5:
            # High similarity: map 0.5-1.0 → 0.7-0.95
            return 0.7 + (best_ngram_score - 0.5) * 0.5
        elif best_ngram_score > 0.3:
            # Medium similarity: map 0.3-0.5 → 0.6-0.7
            return 0.6 + (best_ngram_score - 0.3) * 0.5

        # Strategy 3: Token overlap (fallback, word-level)
        if query_tokens and label_tokens:
            overlap = len(query_tokens & label_tokens)
            # Map to 0.0-0.5 range (strictly below n-gram scores)
            return (overlap / len(query_tokens)) * 0.5

        return 0.2  # Default low score

    def score_components(
        self,
        components: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent = QueryIntent.GENERAL
    ) -> List[Dict[str, Any]]:
        """
        Score all components and add 'relevance_score' field.

        Args:
            components: List of component dicts with 'label' field
            query: User query string
            intent: Query intent for adaptive thresholds (optional)

        Returns:
            Same components list with 'relevance_score' field added
        """
        if not components or not query:
            for comp in components:
                comp['relevance_score'] = 0.2
            return components

        # For small sets, skip scoring (likely all relevant)
        if len(components) <= 10:
            for comp in components:
                comp['relevance_score'] = 0.5
            return components

        # Score each component
        for comp in components:
            label = comp.get('label', '')
            comp['relevance_score'] = self.score_component(label, query)

        # Log results
        threshold = self.INTENT_THRESHOLDS.get(intent, 0.25)
        above_threshold = [c for c in components if c.get('relevance_score', 0) >= threshold]

        print(f"[ComponentScorer] Intent: {intent.value}, Threshold: {threshold:.2f}")
        print(f"[ComponentScorer] {len(above_threshold)}/{len(components)} components above threshold")

        if above_threshold:
            print(f"[ComponentScorer] Top matches:")
            sorted_matches = sorted(
                above_threshold,
                key=lambda x: x.get('relevance_score', 0),
                reverse=True
            )
            for comp in sorted_matches[:3]:
                label = comp.get('label', 'unnamed')
                score = comp.get('relevance_score', 0)
                print(f"  - {label[:60]}: {score:.3f}")

        return components


# Singleton instance
_scorer = None

def get_component_scorer() -> ComponentScorer:
    """Get singleton ComponentScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = ComponentScorer()
    return _scorer
