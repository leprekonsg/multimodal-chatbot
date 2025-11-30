"""
Qwen3-VL Client
Tiered model selection for cost/latency optimization.
Updated with proper bbox_2d format and ASCII logging.
"""
import asyncio
import base64
import json
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

    async def caption_image_structured(
        self,
        image_url: str,
        filename: str = None,
        page_number: int = None
    ) -> Dict[str, Any]:
        """
        Generate structured caption with component bounding boxes for technical manuals.
        
        Uses Qwen3-VL's native bbox_2d format for accurate object detection.
        
        This enables:
        1. Better search indexing (contextual retrieval)
        2. Visual grounding at query time (technicians can locate components)
        3. Precise citations in responses
        
        Returns:
            {
                "description": "Overall page description",
                "transcribed_text": "All visible text",
                "context_prefix": "Contextual retrieval prefix",
                "document_type": "diagram|procedure|table|schematic|other",
                "key_topics": ["topic1", "topic2"],
                "components": [
                    {"label": "component name", "bbox_2d": [x1, y1, x2, y2], "type": "valve|button|..."}
                ]
            }
        
        Note: Qwen3-VL outputs coordinates in NORMALIZED 0-1000 range (not pixels).
        The bbox_2d format is [x1, y1, x2, y2] where each value is 0-1000.
        Frontend must convert: abs_x = (x / 1000) * image_width
        """
        final_url = self._process_image_url(image_url)
        
        if not final_url:
            raise ValueError(f"Could not resolve image: {image_url}")
        
        # Build context string for the prompt
        source_context = ""
        if filename:
            source_context = f"Source: {filename}"
            if page_number:
                source_context += f", Page {page_number}"
        
        # Updated prompt using Qwen3-VL's native bbox_2d format
        prompt = f"""Analyze this technical manual page. {source_context}

You are analyzing a technical document to help technicians find information.

TASK 1 - Detect and locate all important components/elements in the image.
For each component, output its bounding box in this format:
{{"bbox_2d": [x1, y1, x2, y2], "label": "component name"}}

TASK 2 - Provide a structured analysis.

Output your response as a JSON object with this exact format:
{{
    "description": "A detailed description of what this page shows (2-3 sentences)",
    "document_type": "diagram|procedure|table|schematic|specification|other",
    "transcribed_text": "ALL visible text transcribed exactly as shown",
    "key_topics": ["topic1", "topic2", "topic3"],
    "components": [
        {{"bbox_2d": [x1, y1, x2, y2], "label": "component or part name", "type": "button|valve|connector|label|warning|diagram|other"}}
    ]
}}

IMPORTANT RULES:
- Transcribe ALL text exactly as shown (error codes, part numbers, measurements)
- Detect 3-10 most important/identifiable components with their bounding boxes
- For diagrams/schematics, identify key labeled parts, arrows, and annotations
- bbox_2d coordinates should be in NORMALIZED 0-1000 range (Qwen3-VL default format)
- Output ONLY valid JSON, no other text"""

        try:
            # Higher resolution for better component detection
            # Official recommendation: min_pixels=64*32*32, max_pixels=9800*32*32
            response = await self.client.chat.completions.create(
                model=ModelTier.FLASH.value,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url", 
                            "image_url": {"url": final_url},
                            "min_pixels": 64 * 32 * 32,  # ~65K pixels minimum
                            "max_pixels": 2560 * 32 * 32  # ~2.6M pixels maximum (balanced for speed)
                        },
                        {"type": "text", "text": prompt}
                    ]
                }],
                max_tokens=config.qwen.caption_max_tokens + 500  # Extra for JSON + bboxes
            )
            
            if response.usage:
                tracker.track_qwen(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            
            raw_response = response.choices[0].message.content
            
            # Parse JSON response
            try:
                # Handle potential markdown code blocks
                json_text = raw_response
                if "```json" in json_text:
                    json_text = json_text.split("```json")[1].split("```")[0]
                elif "```" in json_text:
                    json_text = json_text.split("```")[1].split("```")[0]
                
                structured = json.loads(json_text.strip())
                
                # Normalize component format (ensure consistent key names)
                components = structured.get("components", [])
                normalized_components = []
                for comp in components:
                    normalized = {
                        "label": comp.get("label", comp.get("name", "Unknown")),
                        "bbox_2d": comp.get("bbox_2d", comp.get("bbox", [])),
                        "type": comp.get("type", "other")
                    }
                    # Only include if we have valid bbox
                    if normalized["bbox_2d"] and len(normalized["bbox_2d"]) == 4:
                        normalized_components.append(normalized)
                
                structured["components"] = normalized_components
                
                # Generate contextual retrieval prefix
                context_prefix = self._generate_context_prefix(
                    structured, filename, page_number
                )
                structured["context_prefix"] = context_prefix
                
                # Log component detection results
                num_components = len(normalized_components)
                if num_components > 0:
                    print(f"[+] Structured caption: {num_components} components detected with bounding boxes")
                    for comp in normalized_components[:3]:
                        print(f"    - {comp['label']}: {comp['bbox_2d']}")
                else:
                    print("[!] Structured caption: No components with valid bounding boxes detected")
                
                return structured
                
            except json.JSONDecodeError as e:
                # Fallback: return raw caption if JSON parsing fails
                print(f"[!] JSON parsing failed ({e}), using plain caption")
                return {
                    "description": raw_response[:500],
                    "transcribed_text": "",
                    "context_prefix": f"[{filename or 'Document'}{f', Page {page_number}' if page_number else ''}] ",
                    "document_type": "other",
                    "key_topics": [],
                    "components": []
                }
            
        except Exception as e:
            print(f"[X] Qwen Structured Caption Error: {e}")
            raise

    def _generate_context_prefix(
        self,
        structured: Dict[str, Any],
        filename: str = None,
        page_number: int = None
    ) -> str:
        """
        Generate contextual retrieval prefix.
        
        Research: Anthropic's contextual retrieval shows 67% error reduction
        by prepending chunk-specific context before embedding.
        
        For technical manuals, we include:
        - Source file and page
        - Document type
        - Key topics
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
                "specification": "Specifications/Parameters"
            }
            parts.append(f"[Type: {type_labels.get(doc_type, doc_type.title())}]")
        
        # Key topics
        topics = structured.get("key_topics", [])
        if topics:
            parts.append(f"[Topics: {', '.join(topics[:3])}]")
        
        return " ".join(parts) + " " if parts else ""

    async def visual_grounding(
        self,
        image_url: str,
        query: str,
        existing_components: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Locate specific elements in an image based on user query.
        
        Uses Qwen3-VL's native visual grounding with bbox_2d format.
        
        This is the query-time visual grounding capability that enables
        technicians to ask "where is the reset button?" and get precise
        bounding box coordinates on the relevant page.
        
        Args:
            image_url: URL or base64 of the source image
            query: User's localization query (e.g., "where is the valve?")
            existing_components: Pre-extracted components from ingestion (optional)
        
        Returns:
            {
                "found": True/False,
                "element": "name of located element",
                "bbox_2d": [x1, y1, x2, y2],
                "description": "explanation of what was found",
                "confidence": "high/medium/low"
            }
        
        Note: Coordinates are in NORMALIZED 0-1000 range (Qwen3-VL default).
        Frontend must convert: abs_x = (x / 1000) * displayed_image_width
        """
        final_url = self._process_image_url(image_url)
        
        if not final_url:
            return {
                "found": False,
                "error": "Could not load image"
            }
        
        # Check if we can use pre-extracted components
        if existing_components:
            # Simple keyword matching against pre-extracted components
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for comp in existing_components:
                comp_label = comp.get("label", comp.get("name", "")).lower()
                comp_words = set(comp_label.split())
                
                # Check for word overlap
                if query_words & comp_words:
                    return {
                        "found": True,
                        "element": comp.get("label", comp.get("name")),
                        "bbox_2d": comp.get("bbox_2d", comp.get("bbox")),
                        "description": f"Found '{comp.get('label', comp.get('name'))}' from pre-indexed components",
                        "confidence": "high",
                        "source": "indexed"
                    }
        
        # Fall back to real-time visual grounding via VLM
        prompt = f"""Locate in this image: "{query}"

If you can find it, respond with JSON:
{{
    "found": true,
    "element": "exact name of what you found",
    "bbox_2d": [x1, y1, x2, y2],
    "description": "brief description of the element and its location",
    "confidence": "high" or "medium" or "low"
}}

If you cannot find it, respond with JSON:
{{
    "found": false,
    "reason": "why it couldn't be found",
    "suggestions": ["alternative things to search for"]
}}

IMPORTANT: bbox_2d coordinates should be in normalized 0-1000 range.
Respond ONLY with valid JSON, no other text."""

        try:
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
            
            try:
                # Clean and parse JSON
                json_text = raw_response
                if "```" in json_text:
                    json_text = json_text.split("```")[1]
                    if json_text.startswith("json"):
                        json_text = json_text[4:]
                    json_text = json_text.split("```")[0]
                
                result = json.loads(json_text.strip())
                result["source"] = "realtime_grounding"
                return result
                
            except json.JSONDecodeError:
                return {
                    "found": False,
                    "error": "Failed to parse grounding response",
                    "raw_response": raw_response[:200]
                }
            
        except Exception as e:
            print(f"[X] Visual Grounding Error: {e}")
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
    
    def _should_use_thinking(self, query: str) -> bool:
        query_lower = query.lower()
        triggers = config.ux.thinking_mode_triggers
        for trigger in triggers:
            if trigger in query_lower:
                return True
        return config.qwen.enable_thinking_default


# Singleton instance
qwen_client = QwenClient()