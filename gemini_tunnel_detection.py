import os
import sys
from typing import List

import google.generativeai as genai

PROMPT = (
    "You are an expert in tunnel safety. "
    "Inspect the provided image for structural anomalies such as cracks, "
    "water leaks, or deformation. Provide a short summary of any issues "
    "that could indicate a problem."
)


def configure_api() -> None:
    """Configure Gemini API using the GOOGLE_API_KEY environment variable."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable")
    genai.configure(api_key=api_key)



def analyze_image(image_path: str, model_name: str) -> str:
    """Return the Gemini analysis result for the given image."""
    model = genai.GenerativeModel(model_name)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    response = model.generate_content([
        PROMPT,
        genai.types.Blob(mime_type="image/jpeg", data=image_bytes),
    ])
    return response.text


def compare_models(models: List[str], image_path: str) -> None:
    """Run the same image through multiple models and print the results."""
    for model in models:
        print(f"\nModel: {model}")
        try:
            result = analyze_image(image_path, model)
            print(result)
        except Exception as exc:
            print(f"Failed to analyze image with {model}: {exc}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python gemini_tunnel_detection.py IMAGE_PATH [MODEL...]")
        sys.exit(1)

    image_path = sys.argv[1]
    models = sys.argv[2:] or ["models/gemini-pro-vision"]

    configure_api()
    compare_models(models, image_path)


if __name__ == "__main__":
    main()
