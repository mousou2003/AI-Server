import fitz  # PyMuPDF
from PIL import Image
import io
import argparse
from pathlib import Path
import base64
import requests
from ollama_manager import OllamaManager

def sanitize_filename(name):
    return "".join(c for c in name if c.isalnum() or c in "._- ").rstrip()

def main():
    ap = argparse.ArgumentParser(description="Convert PDF pages to Markdown using OCR.")
    ap.add_argument("input_pdf", type=Path, help="Input PDF file")
    ap.add_argument("output_md", type=str, nargs="?", default=None, help="Output Markdown file (base name, will be suffixed per page, optional)")
    ap.add_argument("--start", type=int, help="Start page (1-indexed, optional, default: first page)")
    ap.add_argument("--end", type=int, help="End page (1-indexed, exclusive, optional, default: last page)")
    ap.add_argument("--title", type=str, default=None, help="Document title (auto-detect if not set)")
    args = ap.parse_args()

    if not args.input_pdf.exists():
        raise SystemExit(f"Input PDF not found: {args.input_pdf}")

    doc = fitz.open(str(args.input_pdf))
    print(f"[DEBUG] PDF file: {args.input_pdf}")
    print(f"[DEBUG] Number of pages: {len(doc)}")
    try:
        meta = doc.metadata
        print(f"[DEBUG] PDF metadata: {meta}")
    except Exception as e:
        print(f"[DEBUG] Could not read PDF metadata: {e}")

    start = (args.start - 1) if args.start else 0
    end = args.end if args.end else len(doc)
    print(f"[DEBUG] Only processing pages {start+1} to {end}")
    for i in range(start, end):
        if i >= len(doc):
            continue
        page = doc[i]
        rect = page.rect
        print(f"[DEBUG] Processing page {i+1} (size: {rect.width} x {rect.height})")

    title_found = False
    title = args.title or f"Pages_{args.start}_to_{args.end or len(doc)}"


    ollama = OllamaManager()
    model_name = "qwen2.5vl"
    if not ollama.verify_model_exists(model_name):
        print(f"Model {model_name} not found in Ollama. Please pull it first.")
        return

    input_folder = args.input_pdf.parent
    if args.output_md:
        base_name = Path(args.output_md).stem
        ext = Path(args.output_md).suffix or ".md"
    else:
        base_name = args.input_pdf.stem
        ext = ".md"

    for i in range(start, end):
        if i >= len(doc):
            continue
        pix = doc[i].get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        img_b64 = base64.b64encode(img_data).decode("utf-8")
        prompt = "Extract all readable text from this image."
        payload = {
            "model": model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False
        }
        print(f"[DEBUG] Sending OCR request to Ollama: {ollama.config['url']}/api/generate")
        print(f"[DEBUG] Payload: {{'model': {model_name}, 'prompt': {prompt}, 'images': [base64 PNG], 'stream': False}}")
        try:
            response = requests.post(f"{ollama.config['url']}/api/generate", json=payload, timeout=120)
            print(f"[DEBUG] Response status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"[DEBUG] Response JSON: {result}")
                page_text = result.get("response", "")
                preview = page_text.strip()[:200].replace('\n', ' ')
                print(f"[DEBUG] Extracted text preview: {preview if preview else '[empty]'}")
            else:
                print(f"[ERROR] OCR request failed. Status: {response.status_code}, Content: {response.text}")
                page_text = f"[OCR failed: {response.status_code}]"
        except Exception as e:
            print(f"[ERROR] Exception during OCR request: {e}")
            page_text = f"[OCR exception: {e}]"

        if i == start and not title_found and not args.title:
            for line in page_text.splitlines():
                cleaned = line.strip()
                if cleaned:
                    title = sanitize_filename(cleaned[:50])
                    title_found = True
                    break
        out_file = input_folder / f"{base_name}_page_{i+1}{ext}"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n## Page {i + 1}\n\n{page_text.strip()}\n\n")
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
