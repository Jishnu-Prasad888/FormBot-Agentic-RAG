import os
from pathlib import Path
from openai import OpenAI
import base64
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

questions = [
    "What is a realty gold loan in the context of a gold loan application form, and why is it necessary to have builder letters for it?",
    "What are the gross weight and net weight in a gold loan application form, and how should they be entered?",
    "What does 75% LTV mean in relation to a gold loan application form?"
]

output_dir = Path("generated_handwritten_questions")
output_dir.mkdir(exist_ok=True)

for idx, question in enumerate(questions, start=1):
    prompt = f"""
A highly realistic photograph of a slightly off-white notebook paper lying on a desk.

The following text is handwritten naturally with blue ink, realistic human handwriting,
medium neatness, slight imperfections, natural spacing, and authentic pen pressure:

"{question}"

Photorealistic, natural lighting, paper texture visible, realistic shadows,
looks like an actual handwritten note captured by a smartphone camera.
No printed fonts. No digital text. Handwriting only.
Medium image quality.
"""

    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
    )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)

    output_path = output_dir / f"question_{idx}.png"
    with open(output_path, "wb") as f:
        f.write(image_bytes)

    print(f"Saved: {output_path}")

print("Done.")