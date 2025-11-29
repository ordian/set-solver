# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""
Auto-label Set card crops using OpenRouter API with few-shot examples and batching.

Setup:
    1. Create examples directory with 3 reference images:
       dataset/classifier/examples/
         solid.jpg    - a card with solid fill
         striped.jpg  - a card with striped fill
         empty.jpg    - a card with empty fill

    2. Edit EXAMPLES dict below with correct labels for your example images

    3. export OPENROUTER_API_KEY=your_key

Run:
    uv run auto_label.py --limit 10      # Test on 10 images first
    uv run auto_label.py                  # Run on all images
    uv run auto_label.py --batch-size 3   # Adjust batch size

Output:
    dataset/classifier/labels.json
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import requests

# Config
CROPS_DIR = Path("dataset/classifier/crops")
EXAMPLES_DIR = Path("dataset/classifier/examples")
OUTPUT_FILE = Path("dataset/classifier/labels.json")

# ============================================================
# FEW-SHOT EXAMPLES - Edit these to match your example images!
# Put 3 images in EXAMPLES_DIR: solid.jpg, striped.jpg, empty.jpg
# ============================================================
EXAMPLES = {
    "solid.jpg": {
        "count": "one",
        "color": "green",
        "shape": "diamond",
        "fill": "solid",
    },
    "striped.jpg": {
        "count": "three",
        "color": "purple",
        "shape": "squiggle",
        "fill": "striped",
    },
    "empty.jpg": {
        "count": "two",
        "color": "purple",
        "shape": "squiggle",
        "fill": "empty",
    },
}

API_KEY = os.environ["OPENROUTER_API_KEY"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "google/gemini-2.5-flash-image"
# MODEL = "google/gemini-3-pro-image-preview"

SYSTEM_PROMPT = """
You are a Set card classifier. Identify 4 attributes for each card.

## ATTRIBUTES

**COUNT**: one, two, or three shapes

**COLOR**: red, green, or purple

**SHAPE**:
- diamond = pointed at both ends, like ◇
- oval = rounded pill/ellipse shape
- squiggle = wavy curved blob, like a tilde ~ or bean

**FILL** (look carefully INSIDE the shape):
- solid = completely filled with color, no white inside
- striped = has thin parallel lines running through it
- empty = completely white/blank inside, only the outline visible

## CRITICAL: STRIPED VS EMPTY

This is the most common mistake! Striped cards have thin lines inside that are the SAME COLOR as the outline - they can be very subtle and easy to miss.

- If you see ANY lines inside the shape, even faint ones → striped
- If the inside is completely blank/white with NO lines → empty

Respond with ONLY valid JSON (no markdown, no code blocks).
"""


def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def build_example_messages() -> list[dict]:
    """Build few-shot example messages."""
    messages = []

    for filename, label in EXAMPLES.items():
        example_path = EXAMPLES_DIR / filename
        if not example_path.exists():
            print(f"Warning: Example image not found: {example_path}")
            continue

        b64 = encode_image(example_path)

        # User shows image
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this card:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        )

        # Assistant gives correct answer
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(label),
            }
        )

    return messages


def classify_batch(
    image_paths: list[Path], example_messages: list[dict]
) -> list[dict | None]:
    """Classify a batch of card images."""

    # Build content with numbered images
    content = [
        {
            "type": "text",
            "text": f"Classify these {len(image_paths)} cards. Respond with a JSON array of {len(image_paths)} objects in the same order:\n",
        }
    ]

    for i, path in enumerate(image_paths, 1):
        b64 = encode_image(path)
        content.append({"type": "text", "text": f"Card {i}:"})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *example_messages,
        {"role": "user", "content": content},
    ]

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 400 * len(image_paths),
                "reasoning": {"exclude": True},
            },
            timeout=60,
        )

        if response.status_code != 200:
            print(f"HTTP {response.status_code}: {response.text[:200]}")
            return [None] * len(image_paths)

        data = response.json()

        if "error" in data:
            print(f"API error: {data['error']}")
            return [None] * len(image_paths)

        text = data["choices"][0]["message"]["content"].strip()

        # Extract JSON from response (handle markdown code blocks)
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        if "[" in text:
            json_str = text[text.index("[") : text.rindex("]") + 1]
        elif "{" in text:
            json_str = text[text.index("{") : text.rindex("}") + 1]
        else:
            print(f"No JSON in response: {text[:100]}")
            return [None] * len(image_paths)

        parsed = json.loads(json_str)

        # Handle both array and single object responses
        if isinstance(parsed, dict):
            parsed = [parsed]

        return parsed

    except Exception as e:
        print(f"Exception: {e}")
        return [None] * len(image_paths)


def main():
    parser = argparse.ArgumentParser(description="Auto-label Set card crops")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process N images (for testing)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Images per API request (default: 5)"
    )
    args = parser.parse_args()

    print(f"Using model: {MODEL}")
    print(f"Batch size: {args.batch_size}")

    # Load existing labels if any (to resume)
    labels = {}
    if OUTPUT_FILE.exists():
        labels = json.load(open(OUTPUT_FILE))
        print(f"Loaded {len(labels)} existing labels")

    # Build few-shot examples
    print(f"Loading examples from {EXAMPLES_DIR}...")
    example_messages = build_example_messages()
    print(f"Loaded {len(example_messages) // 2} few-shot examples")

    # Get crops to process
    crops = sorted(CROPS_DIR.glob("*.jpg"))
    print(f"Found {len(crops)} crops")

    # Filter already labeled
    crops = [c for c in crops if c.name not in labels]
    print(f"Remaining to label: {len(crops)}")

    if args.limit:
        crops = crops[: args.limit]
        print(f"Limiting to {args.limit} images (test mode)")

    if not crops:
        print("Nothing to do!")
        return

    # Valid values for validation
    valid_colors = {"red", "green", "purple"}
    valid_shapes = {"diamond", "oval", "squiggle"}
    valid_fills = {"solid", "striped", "empty"}
    valid_counts = {"one", "two", "three"}

    # Process in batches
    total_labeled = 0
    for batch_start in range(0, len(crops), args.batch_size):
        batch = crops[batch_start : batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(crops) + args.batch_size - 1) // args.batch_size

        print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} cards...")

        results = classify_batch(batch, example_messages)

        for path, result in zip(batch, results):
            filename = path.name
            if result and (
                result.get("color") in valid_colors
                and result.get("shape") in valid_shapes
                and result.get("fill") in valid_fills
                and result.get("count") in valid_counts
            ):
                labels[filename] = result
                total_labeled += 1
                print(
                    f"  {filename}: {result['count']} {result['color']} {result['fill']} {result['shape']}"
                )
            else:
                print(f"  {filename}: FAILED - {result}")

        # Save after each batch
        OUTPUT_FILE.write_text(json.dumps(labels, indent=2))

        # Small delay between batches
        time.sleep(0.2)

    print(f"\n{'=' * 50}")
    print(f"Done! Labeled {total_labeled}/{len(crops)} cards this run")
    print(f"Total labels: {len(labels)}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
