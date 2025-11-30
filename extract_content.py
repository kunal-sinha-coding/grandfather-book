from openai import AsyncOpenAI
import os
import argparse
import fitz  # PyMuPDF
from tqdm import tqdm
from pathlib import Path
import base64
from dotenv import load_dotenv
import asyncio
from natsort import natsorted

IMAGE_DIR = Path("images")
TEXT_DIR = Path("text")
OUTPUT_FILE = Path("all_output.txt")

SYSTEM_PROMPT = """
Extract the text content from this image.
If you cannot tell what a word is, make an educated guess.
Format the text properly. Make sure to include paragraph breaks when appropriate.
"""

load_dotenv()
openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    organization=os.environ.get("OPENAI_ORGANIZATION")
)

async def extract_content(pdf_path, model_name, image_dpi):
    save_pdf_to_jpegs(pdf_path, image_dpi)
    all_output, total_cost = "", 0
    for image_path in tqdm(natsorted(IMAGE_DIR.glob("*.jpg")), desc="Extracting text from images"):
        extracted_output, cost = await extract_single_page(image_path, model_name)
        save_extracted_output(image_path, extracted_output)
        all_output += f"\n\n{extracted_output}"
        total_cost += cost
        print(f"Total cost: ${total_cost}")
    save_extracted_output(OUTPUT_FILE, all_output)

async def extract_single_page(image_path, model_name):
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    response, cost = "test", 1
    response = await openai_client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                    },
                ],
            }
        ]
    )
    cost = estimate_llm_cost(
        response.usage.input_tokens,
        response.usage.output_tokens
    )
    extracted_output = response.output[0].content[0].text
    return extracted_output, cost

def save_extracted_output(image_path, extracted_output):
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    text_path = TEXT_DIR / Path(image_path.with_suffix(".txt").name)
    text_path.write_text(extracted_output)

def save_pdf_to_jpegs(pdf_path, image_dpi):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    for i, page in tqdm(enumerate(doc), desc="Saving PDF pages as images"):
        # Zoom for DPI
        zoom = image_dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=mat)

        output_path = IMAGE_DIR / f"page_{i+1}.jpg"
        pix.save(str(output_path))  # PyMuPDF requires str()

    doc.close()

def estimate_llm_cost(
    input_tokens,
    output_tokens,
    price_input_per_million=1.25,
    price_output_per_million=10,
) -> float:
    cost_input = (input_tokens / 1_000_000) * price_input_per_million
    cost_output = (output_tokens / 1_000_000) * price_output_per_million
    return cost_input + cost_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf_path",
        type=str,
        help="Path to the input file",
        default="Book1 All JPGs.pdf",

    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Path to the input file",
        default="gpt-5.1"
    )
    parser.add_argument(
        "--image_dpi",
        type=int,
        help="DPI for image resolution",
        default=300
    )
    args = parser.parse_args()
    print(f"PDF path: {args.pdf_path}")
    print(f"Model name: {args.model_name}")
    print(f"Image DPI: {args.image_dpi}")
    asyncio.run(extract_content(args.pdf_path, args.model_name, args.image_dpi))


if __name__ == "__main__":
    main()

