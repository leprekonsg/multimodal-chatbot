"""
Usage tracker with console output for debugging.
"""
from dataclasses import dataclass
import os

# Set to "1" to enable verbose usage logging
VERBOSE_USAGE = os.environ.get("VERBOSE_USAGE", "1") == "1"


@dataclass
class UsageStats:
    voyage_tokens: int = 0
    voyage_pixels: int = 0
    qwen_input_tokens: int = 0
    qwen_output_tokens: int = 0


class UsageTracker:
    def __init__(self):
        self.total = UsageStats()
    
    def track_voyage(self, tokens: int = 0, pixels: int = 0):
        self.total.voyage_tokens += tokens
        self.total.voyage_pixels += pixels
        if VERBOSE_USAGE and (tokens > 0 or pixels > 0):
            print(f"📊 [Voyage] tokens={tokens}, pixels={pixels:,} | Total: {self.total.voyage_tokens} tokens, {self.total.voyage_pixels:,} pixels")
    
    def track_qwen(self, input_tokens: int, output_tokens: int):
        self.total.qwen_input_tokens += input_tokens
        self.total.qwen_output_tokens += output_tokens
        if VERBOSE_USAGE:
            print(f"📊 [Qwen] in={input_tokens}, out={output_tokens} | Total: {self.total.qwen_input_tokens} in, {self.total.qwen_output_tokens} out")
    
    def print_summary(self):
        """Print usage summary and estimated costs."""
        cost = (
            (self.total.voyage_tokens / 1_000_000) * 0.12 +
            (self.total.qwen_input_tokens / 1_000) * 0.004 + 
            (self.total.qwen_output_tokens / 1_000) * 0.012
        )
        print(f"\n{'='*50}")
        print(f"📈 USAGE SUMMARY")
        print(f"{'='*50}")
        print(f"Voyage: {self.total.voyage_tokens:,} tokens, {self.total.voyage_pixels:,} pixels")
        print(f"Qwen:   {self.total.qwen_input_tokens:,} input, {self.total.qwen_output_tokens:,} output tokens")
        print(f"Est. Cost: ${cost:.4f} USD")
        print(f"{'='*50}\n")


tracker = UsageTracker()
