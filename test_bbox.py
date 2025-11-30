#!/usr/bin/env python3
"""
Bounding Box Detection Test Script

Run this to test if Qwen3-VL is returning bounding boxes correctly.
Usage: python test_bbox.py <image_path>

This script will:
1. Send the image to the VLM with the official prompt style
2. Print the raw response for debugging
3. Parse and validate the bounding boxes
4. Optionally render the boxes on the image
"""
import asyncio
import base64
import json
import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_bbox_detection(image_path: str):
    """Test bounding box detection on an image."""
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return
    
    print(f"=" * 60)
    print(f"BOUNDING BOX DETECTION TEST")
    print(f"=" * 60)
    print(f"Image: {image_path}")
    print(f"=" * 60)
    
    # Load and encode image
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    ext = Path(image_path).suffix.lower()
    mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}
    mime = mime_map.get(ext, 'image/jpeg')
    
    b64_image = base64.b64encode(image_data).decode('utf-8')
    data_uri = f"data:{mime};base64,{b64_image}"
    
    print(f"[INFO] Image size: {len(image_data):,} bytes")
    print(f"[INFO] MIME type: {mime}")
    print()
    
    # --- MODIFIED BLOCK START ---
    from openai import AsyncOpenAI
    
    # Default fallback values
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    model_name = "qwen-vl-max" 
    
    # Try loading from project config, but don't crash if it fails
    try:
        from config import config, ModelTier
        if config and hasattr(config, 'qwen'):
            api_key = config.qwen.api_key
            base_url = config.qwen.base_url
            model_name = ModelTier.FLASH.value
    except Exception as e:
        print(f"[WARN] Could not load project config ({e}). Using environment variables fallback.")

    if not api_key:
        print("[ERROR] Missing API Key. Set DASHSCOPE_API_KEY in .env or environment variables.")
        return

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    # --- MODIFIED BLOCK END ---
    
    # Test 1: Simple bbox detection (official style)
    print("=" * 60)
    
    # Test 1: Simple bbox detection (official style)
    print("=" * 60)
    print("TEST 1: Official Style Prompt")
    print("=" * 60)
    
    prompt1 = """Detect all important components and labeled elements in this image.

Report ALL detected items with bounding box coordinates in JSON format:
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "item name"}
]

RULES:
- bbox_2d coordinates in normalized 0-1000 range (NOT pixels)
- x1,y1 = top-left corner, x2,y2 = bottom-right corner  
- Output ONLY the JSON array"""

    print(f"[PROMPT]:\n{prompt1}\n")
    
    try:
        response1 = await client.chat.completions.create(
            model=ModelTier.FLASH.value,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                        "min_pixels": 64 * 32 * 32,
                        "max_pixels": 2560 * 32 * 32
                    },
                    {"type": "text", "text": prompt1}
                ]
            }],
            max_tokens=800
        )
        
        raw1 = response1.choices[0].message.content
        print(f"[RAW RESPONSE]:")
        print("-" * 40)
        print(raw1)
        print("-" * 40)
        
        # Try to parse
        json_text = raw1
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]
        
        try:
            parsed = json.loads(json_text.strip())
            print(f"\n[PARSED JSON]: SUCCESS")
            print(f"[TYPE]: {type(parsed)}")
            
            if isinstance(parsed, list):
                print(f"[COUNT]: {len(parsed)} items")
                for i, item in enumerate(parsed[:10]):
                    bbox = item.get("bbox_2d", item.get("bbox", []))
                    label = item.get("label", item.get("name", "?"))
                    print(f"  [{i+1}] {label}: {bbox}")
            else:
                print(f"[DATA]: {json.dumps(parsed, indent=2)[:500]}")
        except json.JSONDecodeError as e:
            print(f"\n[PARSED JSON]: FAILED - {e}")
            
    except Exception as e:
        print(f"[ERROR]: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test 2: Single object detection
    print("=" * 60)
    print("TEST 2: Single Object Detection (Chinese style)")
    print("=" * 60)
    
    prompt2 = "Locate all text labels in this image, report the bbox coordinates in JSON format."
    
    print(f"[PROMPT]: {prompt2}\n")
    
    try:
        response2 = await client.chat.completions.create(
            model=ModelTier.FLASH.value,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                        "min_pixels": 64 * 32 * 32,
                        "max_pixels": 2560 * 32 * 32
                    },
                    {"type": "text", "text": prompt2}
                ]
            }],
            max_tokens=800
        )
        
        raw2 = response2.choices[0].message.content
        print(f"[RAW RESPONSE]:")
        print("-" * 40)
        print(raw2)
        print("-" * 40)
        
    except Exception as e:
        print(f"[ERROR]: {e}")
    
    print()
    
    # Test 3: Using the fixed llm_client
    print("=" * 60)
    print("TEST 3: Using Fixed QwenClient")
    print("=" * 60)
    
    try:
        # Try to import the fixed client
        from llm_client import qwen_client
        
        components = await qwen_client.detect_bounding_boxes(
            image_url=data_uri,
            target_objects="all labeled components, buttons, text, and important elements"
        )
        
        print(f"[RESULT]: {len(components)} components detected")
        for comp in components:
            print(f"  - {comp['label']}: {comp['bbox_2d']}")
            
    except ImportError:
        print("[SKIP] Fixed llm_client not installed")
    except Exception as e:
        print(f"[ERROR]: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def visualize_boxes(image_path: str, components: list, output_path: str = None):
    """Render bounding boxes on image for visual verification."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.open(image_path)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        
        colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta']
        
        for i, comp in enumerate(components):
            bbox = comp.get("bbox_2d", [])
            label = comp.get("label", f"Item {i+1}")
            color = colors[i % len(colors)]
            
            if len(bbox) == 4:
                # Convert 0-1000 to pixels
                x1 = int(bbox[0] / 1000 * width)
                y1 = int(bbox[1] / 1000 * height)
                x2 = int(bbox[2] / 1000 * width)
                y2 = int(bbox[3] / 1000 * height)
                
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label
                draw.text((x1 + 5, y1 + 5), label, fill=color)
        
        if output_path is None:
            output_path = str(Path(image_path).stem) + "_annotated.png"
        
        img.save(output_path)
        print(f"[OUTPUT] Annotated image saved to: {output_path}")
        
    except ImportError:
        print("[SKIP] PIL not available for visualization")
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_bbox.py <image_path>")
        print("")
        print("This script tests bounding box detection with Qwen3-VL.")
        print("Make sure DASHSCOPE_API_KEY is set in your .env file.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    asyncio.run(test_bbox_detection(image_path))