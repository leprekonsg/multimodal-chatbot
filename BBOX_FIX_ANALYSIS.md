# Qwen3-VL Bounding Box Detection - Analysis & Fix

## Executive Summary

The bounding boxes are not being returned due to **multiple cascading issues** in the prompt design, JSON parsing, and logging. This document details all issues found and the implemented fixes.

---

## Root Cause Analysis

### Issue 1: Complex Multi-Task Prompt (PRIMARY CAUSE)

**Problem:** The current `caption_image_structured` prompt asks for MULTIPLE tasks simultaneously:
- Describe the image
- Classify document type
- Transcribe text
- Extract topics
- **AND** detect bounding boxes

This overloads the model and causes it to prioritize the text generation tasks while skipping or malforming the bounding box detection.

**Evidence from Research:**
> "Unlike GPT-4, which often infers intent, Qwen3-VL requires strict prompting."
> - Bad Prompt: "Where is the cat?"
> - Good Prompt: "Locate the cat. Return the bounding box in JSON format: {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"cat\"}."

**Fix:** Split into two separate API calls:
1. First call: Get description, text, topics
2. Second call: Dedicated bbox detection with simple, focused prompt

---

### Issue 2: Missing Raw Response Logging

**Problem:** When JSON parsing fails or returns no components, the code doesn't log the actual raw response from the VLM. This makes debugging impossible.

**Current Code (llm_client.py:262-263):**
```python
else:
    print("[!] Structured caption: No components with valid bounding boxes detected")
```

We never see WHAT the model actually returned!

**Fix:** Added comprehensive logging:
```python
print(f"[DEBUG] Raw VLM response ({context}):")
print(f"---BEGIN RAW RESPONSE---")
print(raw_response[:2000])
print(f"---END RAW RESPONSE---")
```

---

### Issue 3: Prompt Doesn't Follow Official Style

**Problem:** The current prompt style doesn't match the proven official Qwen3-VL cookbook prompts.

**Current (verbose, complex):**
```
TASK 1 - Detect and locate all important components/elements in the image.
For each component, output its bounding box in this format:
{"bbox_2d": [x1, y1, x2, y2], "label": "component name"}

TASK 2 - Provide a structured analysis...
```

**Official Style (simple, direct):**
```python
# From official Qwen cookbook:
prompt = "Locate the person who act bravely, report the bbox coordinates in JSON format."
```

**Fix:** New dedicated `detect_bounding_boxes()` method uses official style:
```python
prompt = f"""Detect and locate {target_objects} in this image.

Report ALL detected items with their bounding box coordinates in JSON format.

Output format (array of objects):
[
  {{"bbox_2d": [x1, y1, x2, y2], "label": "item name"}}
]

CRITICAL RULES:
- bbox_2d coordinates MUST be in normalized 0-1000 range (NOT pixels)
- Output ONLY the JSON array, no other text
"""
```

---

### Issue 4: Coordinate System Confusion

**Problem:** Qwen3-VL uses **0-1000 normalized coordinates**, not pixel coordinates. The prompt mentions this but isn't explicit enough.

**Research Evidence:**
> "The user, accustomed to Qwen2.5-VL or standard detection libraries, interprets these as absolute pixels. On a 4K image, a box of 100×100 pixels at position (100, 100) is a tiny square in the top-left corner."
> "The Reality: The model meant 'from 10% width to 20% width.'"

**Fix:** Made the coordinate system VERY explicit in the prompt:
```
CRITICAL RULES:
- bbox_2d coordinates MUST be in normalized 0-1000 range (NOT pixels)
- x1,y1 = top-left corner, x2,y2 = bottom-right corner
```

---

### Issue 5: Insufficient max_tokens

**Problem:** The combined prompt asks for a lot of output. If `max_tokens` is too low, the JSON gets truncated, causing parse failures.

**Research Evidence:**
> "The 'Thinking' Interference: If the max_tokens limit is reached during the thinking phase, the structured output is lost or malformed."

**Fix:** 
- Increased token budget for bbox detection
- Split calls so each has adequate tokens

---

### Issue 6: JSON Parsing Not Robust Enough

**Problem:** The current JSON parsing handles `\`\`\`json` blocks but doesn't handle other variations or provide debugging output.

**Fix:** Added `_parse_json_response()` method with multiple strategies:
1. Extract from `\`\`\`json` blocks
2. Extract from plain `\`\`\`` blocks
3. Use regex to find JSON objects/arrays
4. Try to fix truncated JSON (missing closing brackets)
5. Log everything for debugging

---

## Implementation Changes

### New Methods Added to `QwenClient`:

1. **`detect_bounding_boxes()`** - Dedicated bbox detection using official prompt style
2. **`_parse_json_response()`** - Robust JSON parsing with logging

### Modified Methods:

1. **`caption_image_structured()`** - Now uses two-stage approach:
   - Stage 1: Get description/text/topics
   - Stage 2: Call `detect_bounding_boxes()` separately

2. **`visual_grounding()`** - Simplified prompt, better logging

---

## Testing Procedure

Use the provided `test_bbox.py` script:

```bash
# Set your API key
export DASHSCOPE_API_KEY="sk-your-key"

# Test bbox detection
python test_bbox.py /path/to/your/image.png
```

The script will:
1. Send the image with official-style prompts
2. Print RAW responses (critical for debugging!)
3. Try to parse the JSON
4. Report success/failure with details

---

## Expected Output Format

```json
[
  {"bbox_2d": [100, 200, 300, 400], "label": "Reset Button"},
  {"bbox_2d": [500, 100, 700, 250], "label": "Valve A"},
  {"bbox_2d": [50, 500, 200, 600], "label": "Warning Label"}
]
```

**Coordinate Interpretation:**
- `[100, 200, 300, 400]` means:
  - Top-left: 10% from left, 20% from top
  - Bottom-right: 30% from left, 40% from top

**Frontend Conversion:**
```javascript
// Convert 0-1000 to pixels
const absX1 = (bbox[0] / 1000) * imageWidth;
const absY1 = (bbox[1] / 1000) * imageHeight;
const absX2 = (bbox[2] / 1000) * imageWidth;
const absY2 = (bbox[3] / 1000) * imageHeight;
```

---

## Common Pitfalls to Avoid

### 1. Don't combine bbox detection with other tasks
The model gets confused. Keep bbox detection as a separate, focused call.

### 2. Don't expect pixel coordinates
Always remember it's 0-1000 normalized. Convert in frontend/backend.

### 3. Don't skip raw response logging
When debugging, ALWAYS log the actual VLM response before parsing.

### 4. Don't use overly complex JSON structures
Simple `[{bbox_2d, label}]` format works best. Avoid nested structures.

### 5. Don't trust quantized models for bbox detection
> "The 'Blind Model' Syndrome: Users reported that 4B and 8B quantized models (GGUF format) would answer text questions perfectly but fail completely at visual grounding."

Use the official API endpoints with full models.

---

## Verification Checklist

- [ ] Replace `llm_client.py` with the fixed version
- [ ] Run `test_bbox.py` on a sample image
- [ ] Check that raw responses are being logged
- [ ] Verify coordinates are in 0-1000 range
- [ ] Test end-to-end ingestion with a PDF
- [ ] Check that components appear in the Sources panel

---

## Files Modified

1. **`llm_client.py`** - Complete rewrite of bbox-related methods
2. **`test_bbox.py`** - New debug/test script

Copy from `/mnt/user-data/outputs/` to your project directory.
