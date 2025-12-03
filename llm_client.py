"""
Qwen3-VL Client
Tiered model selection for cost/latency optimization.
FIXED: Proper bbox_2d prompts following official Qwen style + comprehensive logging.

Key Fixes:
1. Simplified prompts following official Qwen3-VL cookbook style
2. Added comprehensive raw response logging for debugging
3. Added one-shot examples to improve bbox detection reliability
4. Improved JSON parsing robustness
5. Explicit 0-1000 coordinate system instructions
"""
import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass

from openai import AsyncOpenAI

from config import config, ModelTier
from usage import tracker


@dataclass
class VLMResponse:
    """Response from vision-language model."""
    content: str
    model: str
    thinking_enabled: bool
    input_tokens: int
    output_tokens: int
    latency_ms: float


class QwenClient:
    """
    Async client for Qwen3-VL via Alibaba Cloud ModelStudio (OpenAI Compatible).
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.qwen.api_key,
            base_url=config.qwen.base_url
        )
    
    def _process_image_url(self, url: str) -> Optional[str]:
        """
        Convert localhost URLs to Base64 Data URIs for the API.
        """
        if not url:
            return None
            
        # Check if URL is local
        if "localhost" in url or "127.0.0.1" in url:
            try:
                # Extract filename from URL
                filename = url.split("/")[-1]
                # Use config path
                local_path = Path(config.storage.local_path) / filename
                
                if local_path.exists():
                    with open(local_path, "rb") as f:
                        data = f.read()
                        
                    # Determine mime type
                    ext = local_path.suffix.lower()
                    mime = "image/jpeg"
                    if ext == ".png": mime = "image/png"
                    elif ext == ".webp": mime = "image/webp"
                    elif ext == ".gif": mime = "image/gif"
                    
                    b64 = base64.b64encode(data).decode("utf-8")
                    return f"data:{mime};base64,{b64}"
                else:
                    print(f"[!] File not found at: {local_path}")
                    return None
            except Exception as e:
                print(f"[!] Failed to convert local image {url}: {e}")
                return None
                
        return url

    def _parse_json_response(self, raw_response: str, context: str = "") -> Optional[Dict]:
        """
        Robust JSON parsing with multiple fallback strategies.
        
        Args:
            raw_response: Raw VLM response text
            context: Context string for logging
        
        Returns:
            Parsed JSON dict/list or None
        """
        # Log the raw response for debugging
        print(f"[DEBUG] Raw VLM response ({context}):")
        print(f"---BEGIN RAW RESPONSE---")
        print(raw_response[:2000] if len(raw_response) > 2000 else raw_response)
        if len(raw_response) > 2000:
            print(f"... (truncated, total {len(raw_response)} chars)")
        print(f"---END RAW RESPONSE---")
        
        json_text = raw_response.strip()
        
        # Strategy 0: Try parsing raw response directly first (fastest path)
        try:
            result = json.loads(json_text)
            print(f"[DEBUG] JSON parsing successful (direct)")
            return result
        except json.JSONDecodeError:
            pass  # Continue to extraction strategies
        
        # Strategy 1: Extract from ```json ... ``` blocks
        if "```json" in json_text:
            try:
                extracted = json_text.split("```json")[1].split("```")[0].strip()
                result = json.loads(extracted)
                print(f"[DEBUG] Extracted and parsed JSON from ```json block")
                return result
            except (IndexError, json.JSONDecodeError):
                pass
        
        # Strategy 2: Extract from plain ``` blocks
        if "```" in json_text:
            try:
                parts = json_text.split("```")
                if len(parts) >= 2:
                    extracted = parts[1]
                    if extracted.startswith("json"):
                        extracted = extracted[4:]
                    extracted = extracted.strip()
                    result = json.loads(extracted)
                    print(f"[DEBUG] Extracted and parsed JSON from ``` block")
                    return result
            except (IndexError, json.JSONDecodeError):
                pass
        
        # Strategy 3: Find JSON array with regex FIRST (for bbox responses)
        # This is critical: check for arrays BEFORE objects because bbox responses are arrays
        array_match = re.search(r'\[[\s\S]*\]', json_text)
        if array_match:
            try:
                extracted = array_match.group(0)
                result = json.loads(extracted)
                print(f"[DEBUG] Extracted and parsed JSON array via regex")
                return result
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Find JSON object with regex
        json_match = re.search(r'\{[\s\S]*\}', json_text)
        if json_match:
            try:
                extracted = json_match.group(0)
                result = json.loads(extracted)
                print(f"[DEBUG] Extracted and parsed JSON object via regex")
                return result
            except json.JSONDecodeError as e:
                print(f"[!] JSON object regex match failed to parse: {e}")
        
        # Strategy 5: Try to fix truncated JSON
        # Find the best candidate (array or object)
        candidates = []
        if array_match:
            candidates.append(array_match.group(0))
        if json_match:
            candidates.append(json_match.group(0))
        
        for candidate in candidates:
            try:
                fixed = candidate
                # Fix missing closing brackets
                open_braces = fixed.count('{')
                close_braces = fixed.count('}')
                open_brackets = fixed.count('[')
                close_brackets = fixed.count(']')
                
                if open_braces > close_braces:
                    fixed += '}' * (open_braces - close_braces)
                if open_brackets > close_brackets:
                    fixed += ']' * (open_brackets - close_brackets)
                
                result = json.loads(fixed)
                print(f"[DEBUG] JSON parsing successful after bracket fix")
                return result
            except json.JSONDecodeError:
                continue
        
        print(f"[!] All JSON parsing strategies failed")
        print(f"[!] Final text attempted: {json_text[:300]}...")
        return None

    async def caption_image(
        self,
        image_url: str,
        detail_level: str = "high"
    ) -> str:
        """Generate detailed caption for image using Qwen3-VL-Flash."""
        # Handle local images
        final_url = self._process_image_url(image_url)
        
        if not final_url:
            raise ValueError(f"Could not resolve local image path for: {image_url}")
        
        if detail_level == "high":
            prompt = """Describe this image for a search engine. 
Transcribe ALL visible text, numbers, labels, and axis titles exactly as shown.
Describe colors, spatial layout, and relationships between elements.
If this is a chart/diagram, describe the data patterns and trends."""
        else:
            prompt = "Describe this image concisely, noting key objects and text."
        
        try:
            response = await self.client.chat.completions.create(
                model=ModelTier.FLASH.value,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": final_url}},
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_tokens=config.qwen.caption_max_tokens
            )
            
            if response.usage:
                tracker.track_qwen(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[X] Qwen Caption Error: {e}")
            raise

    async def detect_bounding_boxes(
        self,
        image_url: str,
        target_objects: str = "all important components and labeled elements"
    ) -> List[Dict]:
        """
        Dedicated bounding box detection using official Qwen3-VL prompt style.
        
        This follows the official cookbook style which is proven to work:
        "Locate X, report the bbox coordinates in JSON format."
        
        Args:
            image_url: Image URL or base64 data URI
            target_objects: Description of what to detect
        
        Returns:
            List of {"bbox_2d": [x1,y1,x2,y2], "label": "name"} dicts
            Coordinates are in 0-1000 normalized range.
        """
        final_url = self._process_image_url(image_url)
        
        if not final_url:
            print(f"[!] Could not resolve image for bbox detection")
            return []
        
        # OFFICIAL STYLE PROMPT - Simple and direct
        # Based on: "Locate X, report the bbox coordinates in JSON format."
        prompt = f"""Detect and locate {target_objects} in this image.

Report ALL detected items with their bounding box coordinates in JSON format.

Output format (array of objects):
[
  {{"bbox_2d": [x1, y1, x2, y2], "label": "item name"}},
  {{"bbox_2d": [x1, y1, x2, y2], "label": "another item"}}
]

CRITICAL RULES:
- bbox_2d coordinates MUST be in normalized 0-1000 range (NOT pixels)
- x1,y1 = top-left corner, x2,y2 = bottom-right corner
- Output ONLY the JSON array, no other text
- Detect 3-10 most prominent/important items"""

        try:
            print(f"[DEBUG] Sending bbox detection request...")
            
            response = await self.client.chat.completions.create(
                model=ModelTier.FLASH.value,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url", 
                            "image_url": {"url": final_url},
                            "min_pixels": 64 * 32 * 32,
                            "max_pixels": 2560 * 32 * 32
                        },
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_tokens=800  # Enough for ~10 bboxes
            )
            
            if response.usage:
                tracker.track_qwen(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            
            raw_response = response.choices[0].message.content
            
            # Parse the response
            parsed = self._parse_json_response(raw_response, "bbox_detection")
            
            if parsed is None:
                print(f"[!] Bbox detection: Failed to parse response")
                return []
            
            # Ensure it's a list
            if isinstance(parsed, dict):
                # Sometimes model returns {"components": [...]} or similar
                for key in ["components", "items", "objects", "detections", "results"]:
                    if key in parsed and isinstance(parsed[key], list):
                        parsed = parsed[key]
                        break
                else:
                    # Single detection, wrap in list
                    if "bbox_2d" in parsed:
                        parsed = [parsed]
                    else:
                        print(f"[!] Unexpected response structure: {list(parsed.keys())}")
                        return []
            
            if not isinstance(parsed, list):
                print(f"[!] Expected list, got {type(parsed)}")
                return []
            
            # Validate and normalize components
            valid_components = []
            for comp in parsed:
                if not isinstance(comp, dict):
                    continue
                    
                bbox = comp.get("bbox_2d", comp.get("bbox", comp.get("box", [])))
                label = comp.get("label", comp.get("name", "Unknown"))
                
                # Validate bbox format
                if bbox and len(bbox) == 4:
                    try:
                        # Ensure numeric values
                        bbox = [float(x) for x in bbox]
                        
                        # Validate range (should be 0-1000)
                        if all(0 <= x <= 1000 for x in bbox):
                            valid_components.append({
                                "bbox_2d": [int(x) for x in bbox],
                                "label": str(label),
                                "type": comp.get("type", "other")
                            })
                        else:
                            print(f"[!] Bbox out of 0-1000 range: {bbox} for '{label}'")
                    except (ValueError, TypeError) as e:
                        print(f"[!] Invalid bbox values: {bbox}, error: {e}")
            
            print(f"[+] Bbox detection: Found {len(valid_components)} valid components")
            for comp in valid_components[:5]:
                print(f"    - {comp['label']}: {comp['bbox_2d']}")
            
            return valid_components
            
        except Exception as e:
            print(f"[X] Bbox Detection Error: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def caption_image_structured(
        self,
        image_url: str,
        filename: str = None,
        page_number: int = None
    ) -> Dict[str, Any]:
        """
        Generate structured caption with component bounding boxes for technical manuals.
        
        FIXED: Now uses a two-stage approach:
        1. Get description and text transcription
        2. Separately detect bounding boxes (more reliable)
        
        Returns:
            {
                "description": "Overall page description",
                "transcribed_text": "All visible text",
                "context_prefix": "Contextual retrieval prefix",
                "document_type": "diagram|procedure|table|schematic|other",
                "key_topics": ["topic1", "topic2"],
                "components": [
                    {"label": "component name", "bbox_2d": [x1, y1, x2, y2], "type": "..."}
                ]
            }
        """
        final_url = self._process_image_url(image_url)
        
        if not final_url:
            raise ValueError(f"Could not resolve image: {image_url}")
        
        # Build context string
        source_context = ""
        if filename:
            source_context = f"Source: {filename}"
            if page_number:
                source_context += f", Page {page_number}"
        
        print(f"[+] Structured caption request for: {source_context or 'unnamed image'}")
        
        # STAGE 1: Get description and transcription
        # Use a simpler prompt for better JSON compliance
        description_prompt = f"""Analyze this image. {source_context}

Output a JSON object with this exact format:
{{
    "description": "A detailed 2-3 sentence description of what this image shows",
    "document_type": "diagram|procedure|table|schematic|specification|photo|other",
    "transcribed_text": "ALL visible text transcribed exactly as shown",
    "key_topics": ["topic1", "topic2", "topic3"]
}}

Rules:
- Transcribe ALL visible text exactly (part numbers, labels, measurements)
- Choose the most appropriate document_type
- List 2-5 key topics covered
- Output ONLY valid JSON"""

        try:
            print(f"[DEBUG] Stage 1: Getting description...")
            
            response = await self.client.chat.completions.create(
                model=ModelTier.FLASH.value,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url", 
                            "image_url": {"url": final_url},
                            "min_pixels": 64 * 32 * 32,
                            "max_pixels": 2560 * 32 * 32
                        },
                        {"type": "text", "text": description_prompt}
                    ]
                }],
                max_tokens=config.qwen.caption_max_tokens
            )
            
            if response.usage:
                tracker.track_qwen(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            
            raw_response = response.choices[0].message.content
            
            # Parse description response
            structured = self._parse_json_response(raw_response, "description")
            
            if structured is None:
                print(f"[!] Failed to parse description, using fallback")
                structured = {
                    "description": raw_response[:500],
                    "transcribed_text": "",
                    "document_type": "other",
                    "key_topics": []
                }
            
            # STAGE 2: Get bounding boxes separately
            print(f"[DEBUG] Stage 2: Detecting bounding boxes...")
            components = await self.detect_bounding_boxes(
                image_url=image_url,
                target_objects="all labeled components, buttons, valves, connectors, diagrams, text labels, and important visual elements"
            )
            
            structured["components"] = components
            
            # Generate contextual retrieval prefix
            context_prefix = self._generate_context_prefix(
                structured, filename, page_number
            )
            structured["context_prefix"] = context_prefix
            
            # Log results summary
            num_components = len(components)
            print(f"[+] Structured caption complete:")
            print(f"    - Description: {structured.get('description', '')[:100]}...")
            print(f"    - Document type: {structured.get('document_type', 'unknown')}")
            print(f"    - Components: {num_components} with bounding boxes")
            print(f"    - Topics: {structured.get('key_topics', [])}")
            
            return structured
            
        except Exception as e:
            print(f"[X] Qwen Structured Caption Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _generate_context_prefix(
        self,
        structured: Dict[str, Any],
        filename: str = None,
        page_number: int = None
    ) -> str:
        """
        Generate enriched contextual retrieval prefix.

        Research: Anthropic's contextual retrieval shows 67% error reduction
        by prepending chunk-specific context before embedding.

        Enhanced to include component names for semantic label search,
        enabling text_dense embeddings to match component-based queries
        without requiring a separate lightweight model.
        """
        parts = []

        # Source identification
        if filename:
            source = filename.replace("_", " ").replace("-", " ")
            if page_number:
                source += f", Page {page_number}"
            parts.append(f"[Source: {source}]")

        # Document type
        doc_type = structured.get("document_type", "").lower()
        if doc_type and doc_type != "other":
            type_labels = {
                "diagram": "Technical Diagram",
                "procedure": "Procedure/Instructions",
                "table": "Data Table/Specifications",
                "schematic": "Schematic/Wiring Diagram",
                "specification": "Specifications/Parameters",
                "photo": "Photograph"
            }
            parts.append(f"[Type: {type_labels.get(doc_type, doc_type.title())}]")

        # Key topics
        topics = structured.get("key_topics", [])
        if topics:
            parts.append(f"[Topics: {', '.join(topics[:3])}]")

        # Component names (CRITICAL for semantic label search)
        # This enables queries like "find pressure valve" to match via text_dense embeddings
        components = structured.get("components", [])
        if components:
            # Extract unique component labels, limit to first 10 to avoid token bloat
            component_labels = [
                comp.get("label", comp.get("name", "")).strip()
                for comp in components[:10]
                if comp.get("label") or comp.get("name")
            ]
            if component_labels:
                parts.append(f"[Components: {', '.join(component_labels)}]")

        return " ".join(parts) + " " if parts else ""

    async def visual_grounding(
        self,
        image_url: str,
        query: str,
        existing_components: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Locate specific elements in an image based on user query.
        
        FIXED: Uses official Qwen3-VL prompt style for better reliability.
        
        Args:
            image_url: URL or base64 of the source image
            query: User's localization query (e.g., "where is the valve?")
            existing_components: Pre-extracted components from ingestion (optional)
        
        Returns:
            {
                "found": True/False,
                "element": "name of located element",
                "bbox_2d": [x1, y1, x2, y2],  # 0-1000 normalized
                "description": "explanation of what was found",
                "confidence": "high/medium/low"
            }
        """
        final_url = self._process_image_url(image_url)
        
        if not final_url:
            return {
                "found": False,
                "error": "Could not load image"
            }
        
        # Check if we can use pre-extracted components first
        if existing_components:
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            # Remove common words that don't help matching
            stopwords = {'the', 'a', 'an', 'is', 'are', 'where', 'find', 'locate', 'show', 'me'}
            query_words = query_words - stopwords
            
            for comp in existing_components:
                comp_label = comp.get("label", comp.get("name", "")).lower()
                comp_words = set(comp_label.split())
                
                # Check for word overlap
                if query_words & comp_words:
                    bbox = comp.get("bbox_2d", comp.get("bbox"))
                    if bbox and len(bbox) == 4:
                        print(f"[+] Visual grounding: Found '{comp_label}' from indexed components")
                        print(f"    Bbox: {bbox}")
                        return {
                            "found": True,
                            "element": comp.get("label", comp.get("name")),
                            "bbox_2d": bbox,
                            "description": f"Found '{comp.get('label', comp.get('name'))}' from pre-indexed components",
                            "confidence": "high",
                            "source": "indexed"
                        }
        
        # Fall back to real-time visual grounding via VLM
        # OFFICIAL STYLE PROMPT - Simple and direct
        prompt = f"""Locate "{query}" in this image and report its bounding box.

Output JSON format:
{{
    "found": true,
    "element": "exact name of what you found",
    "bbox_2d": [x1, y1, x2, y2],
    "confidence": "high"
}}

If NOT found:
{{
    "found": false,
    "reason": "brief explanation"
}}

RULES:
- bbox_2d coordinates in normalized 0-1000 range (NOT pixels)
- x1,y1 = top-left, x2,y2 = bottom-right
- Output ONLY valid JSON"""

        try:
            print(f"[DEBUG] Visual grounding request for: '{query}'")
            
            response = await self.client.chat.completions.create(
                model=ModelTier.PLUS.value,  # Use Plus for better grounding
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url", 
                            "image_url": {"url": final_url},
                            "min_pixels": 64 * 32 * 32,
                            "max_pixels": 2560 * 32 * 32
                        },
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_tokens=300
            )
            
            if response.usage:
                tracker.track_qwen(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            
            raw_response = response.choices[0].message.content
            
            # Parse response
            result = self._parse_json_response(raw_response, "visual_grounding")
            
            if result is None:
                return {
                    "found": False,
                    "error": "Failed to parse grounding response",
                    "raw_response": raw_response[:200]
                }
            
            result["source"] = "realtime_grounding"
            
            # Log the result
            if result.get("found"):
                print(f"[+] Visual grounding: Found '{result.get('element')}'")
                print(f"    Bbox: {result.get('bbox_2d')}")
            else:
                print(f"[!] Visual grounding: Not found - {result.get('reason', 'unknown')}")
            
            return result
            
        except Exception as e:
            print(f"[X] Visual Grounding Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "found": False,
                "error": str(e)
            }

    async def generate_response(
        self,
        query: str,
        context: str,
        image_urls: List[str] = None,
        user_image_url: str = None,
        enable_thinking: bool = None
    ) -> VLMResponse:
        """Generate RAG response using Qwen3-VL-Plus."""
        import time
        start_time = time.time()
        
        content = []
        
        # Add user's uploaded image first (if any)
        if user_image_url:
            final_user_url = self._process_image_url(user_image_url)
            if final_user_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": final_user_url}
                })
        
        # Add retrieved images (Qwen3-VL supports multi-image, use up to 5)
        if image_urls:
            for url in image_urls[:5]:
                final_url = self._process_image_url(url)
                if final_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": final_url}
                    })
        
        system_prompt = """You are a helpful assistant answering questions based on a knowledge base.
Use the provided context and images to answer accurately.
If the context doesn't contain enough information, clearly state what's missing.
When referencing images, mention which source image contains the information."""

        user_prompt = f"""Based on the following knowledge base context, answer the user's question.

CONTEXT:
{context}

USER QUESTION: {query}

Instructions:
1. Answer based ONLY on the provided context and images
2. If information is missing, say "I don't have enough information about [topic]"
3. Cite which source (e.g., "According to [image/document name]...") when possible"""
        
        content.append({"type": "text", "text": user_prompt})
        
        if enable_thinking is None:
            enable_thinking = self._should_use_thinking(query)
        
        # Qwen3-VL specific thinking parameters
        extra_body = {}
        if enable_thinking:
            extra_body["enable_thinking"] = True 
        
        try:
            response = await self.client.chat.completions.create(
                model=ModelTier.PLUS.value,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=config.qwen.max_output_tokens,
                extra_body=extra_body if extra_body else None
            )
            
            if response.usage:
                tracker.track_qwen(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
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
    
    async def generate_response_stream(
        self,
        query: str,
        context: str,
        image_urls: List[str] = None,
        user_image_url: str = None,
        enable_thinking: bool = None
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens."""
        content = []
        
        if user_image_url:
            final_user_url = self._process_image_url(user_image_url)
            if final_user_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": final_user_url}
                })
        
        if image_urls:
            for url in image_urls[:5]:
                final_url = self._process_image_url(url)
                if final_url:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": final_url}
                    })
        
        user_prompt = f"""Based on the following context, answer the question.

CONTEXT:
{context}

QUESTION: {query}

Answer based on the context. If information is missing, clearly state that."""
        
        content.append({"type": "text", "text": user_prompt})
        
        if enable_thinking is None:
            enable_thinking = self._should_use_thinking(query)

        extra_body = {}
        if enable_thinking:
            extra_body["enable_thinking"] = True
        
        try:
            stream = await self.client.chat.completions.create(
                model=ModelTier.PLUS.value,
                messages=[{"role": "user", "content": content}],
                max_tokens=config.qwen.max_output_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body=extra_body if extra_body else None
            )
            
            async for chunk in stream:
                if chunk.usage:
                    tracker.track_qwen(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens
                    )
                
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"[X] Qwen Stream Error: {e}")
            raise
    
    async def analyze_sentiment(self, text: str) -> float:
        try:
            response = await self.client.chat.completions.create(
                model=ModelTier.TURBO.value,
                messages=[
                    {"role": "system", "content": "Analyze sentiment. Return ONLY a float between -1.0 and 1.0."},
                    {"role": "user", "content": text}
                ],
                max_tokens=10
            )
            
            if response.usage:
                tracker.track_qwen(response.usage.prompt_tokens, response.usage.completion_tokens)
                
            score_text = response.choices[0].message.content.strip()
            return float(score_text)
        except (ValueError, AttributeError):
            return 0.0
    
    async def rewrite_query(self, history: List[Dict[str, str]], current_query: str) -> str:
        if not history:
            return current_query

        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content'][:200]}"
            for msg in history[-4:]
        ])

        prompt = f"""Chat History:
{history_text}

User's Latest Query: {current_query}

Rewrite the query to be standalone. Return ONLY the rewritten query."""

        try:
            response = await self.client.chat.completions.create(
                model=ModelTier.TURBO.value,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )

            if response.usage:
                tracker.track_qwen(response.usage.prompt_tokens, response.usage.completion_tokens)

            rewritten = response.choices[0].message.content.strip()
            if len(rewritten) > len(current_query) * 3:
                return current_query
            return rewritten
        except Exception:
            return current_query

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
        # Note: Caller (chatbot.py) already limits to max_kb_images (3)
        if retrieved_image_urls:
            for url in retrieved_image_urls:
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

    def _should_use_thinking(self, query: str) -> bool:
        query_lower = query.lower()
        triggers = config.ux.thinking_mode_triggers
        for trigger in triggers:
            if trigger in query_lower:
                return True
        return config.qwen.enable_thinking_default


# Singleton instance
qwen_client = QwenClient()