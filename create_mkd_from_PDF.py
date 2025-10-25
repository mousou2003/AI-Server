import fitz  # PyMuPDF
from PIL import Image
import io
import argparse
from pathlib import Path
import base64
import requests
from ollama_manager import OllamaManager
import re

def sanitize_filename(name):
    return "".join(c for c in name if c.isalnum() or c in "._- ").rstrip()

def main():
    ap = argparse.ArgumentParser(description="Convert PDFs or images to Markdown using OCR.")
    ap.add_argument("input_pdf", type=Path, help="Input PDF file/folder, image file, or folder of images; also folder of Markdown files when using --combine-md")
    ap.add_argument("output_md", type=str, nargs="?", default=None, help="When converting a single PDF without --combined, this is the base name for per-page files. When --combined is set, this is the single output Markdown file path. Optional.")
    ap.add_argument("--start", type=int, help="Start page (1-indexed, optional, default: first page)")
    ap.add_argument("--end", type=int, help="End page (1-indexed, exclusive, optional, default: last page)")
    ap.add_argument("--title", type=str, default=None, help="Document title (auto-detect if not set)")
    ap.add_argument("--combined", action="store_true", help="Combine output into a single Markdown file. If input is a folder, matching files will be processed and merged in order.")
    ap.add_argument("--ext", type=str, default="pdf", help="File extension to include when input is a folder (pdf, png, jpg, jpeg, webp, bmp, tiff; default: pdf)")
    # New: Combine existing Markdown files
    ap.add_argument("--combine-md", action="store_true", help="Combine multiple Markdown files from the input folder into a single Markdown file.")
    ap.add_argument("--md-ext", type=str, default="md", help="File extension for Markdown files when using --combine-md (default: md)")
    ap.add_argument("--no-file-headers", action="store_true", help="When using --combine-md, do not insert per-file headers in the combined output.")
    args = ap.parse_args()

    if not args.input_pdf.exists():
        raise SystemExit(f"Input path not found: {args.input_pdf}")

    # Helper: natural sort using page number in filename when present
    def _page_sort_key(p: Path, page_hint: str = "page"):
        stem = p.stem.lower()
        # Prefer explicit 'page' pattern
        m = re.search(rf"{page_hint}[_\- ]*(\d+)", stem, re.IGNORECASE)
        if m:
            base = re.sub(rf"[_\- ]*{page_hint}[_\- ]*\d+", "", stem, flags=re.IGNORECASE)
            try:
                num = int(m.group(1))
            except Exception:
                num = float('inf')
            return (base, num, stem)
        # Fallback: any trailing/in-file number
        nums = re.findall(r"\d+", stem)
        num = int(nums[-1]) if nums else float('inf')
        base2 = re.sub(r"\d+", "", stem) if nums else stem
        return (base2, num, stem)

    # If combining existing Markdown files, handle here and exit
    if args.combine_md:
        if not args.input_pdf.is_dir():
            raise SystemExit("When using --combine-md, provide a folder path containing Markdown files.")
        folder = args.input_pdf
        # Determine combined output path first to avoid including it
        if args.output_md:
            combined_out = Path(args.output_md)
            if not combined_out.suffix:
                combined_out = combined_out.with_suffix(".md")
        else:
            combined_out = folder / f"{folder.stem}.md"

        mdfiles = sorted([p for p in folder.glob(f"*.{args.md_ext}") if p.is_file()], key=lambda p: _page_sort_key(p, page_hint="page"))
        # Exclude the output file itself if it matches extension and is in same folder
        mdfiles = [p for p in mdfiles if p.resolve() != combined_out.resolve()]
        if not mdfiles:
            raise SystemExit(f"No *.{args.md_ext} files found in: {folder}")

        lines: list[str] = [f"# Combined Markdown from {folder.stem}"]
        for fpath in mdfiles:
            print(f"[DEBUG] Including markdown file: {fpath}")
            try:
                content = fpath.read_text(encoding="utf-8")
            except Exception as e:
                print(f"[ERROR] Could not read {fpath}: {e}")
                content = f"[Read error: {e}]"
            if not args.no_file_headers:
                lines.append(f"\n\n## File: {fpath.stem}\n")
            lines.append(content.strip())

        combined_out.parent.mkdir(parents=True, exist_ok=True)
        combined_out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"Saved combined Markdown: {combined_out}")
        return

    # Common OCR setup for images and PDFs
    image_exts = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}

    def process_image(img_path: Path, combined_lines: list | None = None):
        print(f"[DEBUG] Image file: {img_path}")
        try:
            img_bytes = img_path.read_bytes()
        except Exception as e:
            print(f"[ERROR] Could not read image {img_path}: {e}")
            img_bytes = None

        local_title = args.title or sanitize_filename(img_path.stem)

        ollama = OllamaManager()
        model_name = "qwen2.5vl"
        if not ollama.verify_model_exists(model_name):
            print(f"Model {model_name} not found in Ollama. Please pull it first.")
            return

        if img_bytes is None:
            page_text = f"[Read error: {img_path.name}]"
        else:
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            prompt = "Extract all readable text from this image."
            print(f"[DEBUG] Sending OCR request to Ollama (streaming mode): {ollama.config['url']}/api/generate")
            try:
                text_chunks = []
                for chunk in ollama.stream_inference(model_name, prompt, images=[img_b64], timeout=600):
                    text_chunks.append(chunk)
                page_text = "".join(text_chunks)
                preview = page_text.strip()[:200].replace('\n', ' ')
                print(f"[DEBUG] Extracted text preview: {preview if preview else '[empty]'}")
            except Exception as e:
                print(f"[ERROR] Streaming OCR request failed: {e}")
                page_text = f"[OCR streaming error: {e}]"

        if args.combined and combined_lines is not None:
            combined_lines.append(f"\n\n## Image: {img_path.stem}\n")
            combined_lines.append(page_text.strip())
        else:
            out_file = img_path.with_suffix("").parent / f"{img_path.stem}.md"
            # If output_md specified for single image, honor it
            if args.output_md and not args.input_pdf.is_dir():
                out_file = Path(args.output_md)
                if not out_file.suffix:
                    out_file = out_file.with_suffix(".md")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"# {local_title}\n\n{page_text.strip()}\n")
            print(f"Saved: {out_file}")
    def process_pdf(pdf_path: Path, combined_lines: list | None = None):
        doc = fitz.open(str(pdf_path))
        print(f"[DEBUG] PDF file: {pdf_path}")
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
        local_title = args.title or f"Pages_{args.start}_to_{args.end or len(doc)}"

        # Set up Ollama OCR model
        ollama = OllamaManager()
        model_name = "qwen2.5vl"
        if not ollama.verify_model_exists(model_name):
            print(f"Model {model_name} not found in Ollama. Please pull it first.")
            return

        input_folder = pdf_path.parent
        if args.output_md:
            base_name = Path(args.output_md).stem
            ext = Path(args.output_md).suffix or ".md"
        else:
            base_name = pdf_path.stem
            ext = ".md"

        # If we're combining, add a single header per document
        if args.combined and combined_lines is not None:
            combined_lines.append(f"\n\n## Document: {pdf_path.stem}")

        # If we're combining, we'll append to combined_lines; otherwise, write per-page files
        for i in range(start, end):
            if i >= len(doc):
                continue
            pix = doc[i].get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            img_b64 = base64.b64encode(img_data).decode("utf-8")
            prompt = "Extract all readable text from this image."
            print(f"[DEBUG] Sending OCR request to Ollama (streaming mode): {ollama.config['url']}/api/generate")
            try:
                text_chunks = []
                for chunk in ollama.stream_inference(model_name, prompt, images=[img_b64], timeout=600):
                    text_chunks.append(chunk)
                page_text = "".join(text_chunks)
                preview = page_text.strip()[:200].replace('\n', ' ')
                print(f"[DEBUG] Extracted text preview: {preview if preview else '[empty]'}")
            except Exception as e:
                print(f"[ERROR] Streaming OCR request failed: {e}")
                page_text = f"[OCR streaming error: {e}]"

            if i == start and not title_found and not args.title:
                for line in page_text.splitlines():
                    cleaned = line.strip()
                    if cleaned:
                        local_title = sanitize_filename(cleaned[:50])
                        title_found = True
                        break

            if args.combined and combined_lines is not None:
                # Append to combined content (page-level within current document)
                combined_lines.append(f"### Page {i + 1}\n\n{page_text.strip()}\n")
            else:
                # Write per-page files (original behavior)
                out_file = input_folder / f"{base_name}_page_{i+1}{ext}"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(f"# {local_title}\n\n## Page {i + 1}\n\n{page_text.strip()}\n\n")
                print(f"Saved: {out_file}")

    # Execution flow: single PDF or folder of PDFs
    if args.input_pdf.is_dir():
        folder = args.input_pdf
        ext_lower = args.ext.lower()
        files = sorted([p for p in folder.glob(f"*.{ext_lower}") if p.is_file()], key=lambda p: _page_sort_key(p, page_hint="page"))
        if not files:
            raise SystemExit(f"No *.{args.ext} files found in: {folder}")

        if args.combined:
            # Determine output markdown path
            if args.output_md:
                combined_out = Path(args.output_md)
                if not combined_out.suffix:
                    combined_out = combined_out.with_suffix(".md")
            else:
                combined_out = folder / f"{folder.stem}.md"

            combined_lines: list[str] = [f"# Combined OCR for folder: {folder.stem}"]
            for fp in files:
                print(f"[DEBUG] Processing file in folder: {fp}")
                if ext_lower in image_exts:
                    process_image(fp, combined_lines=combined_lines)
                elif ext_lower == "pdf":
                    process_pdf(fp, combined_lines=combined_lines)
                else:
                    print(f"[WARN] Skipping unsupported extension: {fp.suffix}")
            # Write combined file once
            combined_out.parent.mkdir(parents=True, exist_ok=True)
            with open(combined_out, "w", encoding="utf-8") as f:
                f.write("\n".join(combined_lines).rstrip() + "\n")
            print(f"Saved combined Markdown: {combined_out}")
        else:
            # Per-file outputs in-place
            for fp in files:
                if ext_lower in image_exts:
                    process_image(fp, combined_lines=None)
                elif ext_lower == "pdf":
                    process_pdf(fp, combined_lines=None)
                else:
                    print(f"[WARN] Skipping unsupported extension: {fp.suffix}")
    else:
        # Single PDF path
        suffix = args.input_pdf.suffix.lower().lstrip('.')
        if suffix in image_exts:
            if args.combined:
                # Combined output (single image still allowed)
                combined_out = Path(args.output_md) if args.output_md else args.input_pdf.with_suffix(".md")
                if not combined_out.suffix:
                    combined_out = combined_out.with_suffix(".md")
                combined_lines: list[str] = [f"# {args.title or sanitize_filename(args.input_pdf.stem)}"]
                process_image(args.input_pdf, combined_lines=combined_lines)
                with open(combined_out, "w", encoding="utf-8") as f:
                    f.write("\n".join(combined_lines).rstrip() + "\n")
                print(f"Saved combined Markdown: {combined_out}")
            else:
                process_image(args.input_pdf, combined_lines=None)
        else:
            # Treat as PDF
            if args.combined:
                # One markdown with all pages of this PDF
                if args.output_md:
                    combined_out = Path(args.output_md)
                    if not combined_out.suffix:
                        combined_out = combined_out.with_suffix(".md")
                else:
                    combined_out = args.input_pdf.with_suffix(".md")
                combined_lines: list[str] = [f"# {args.title or sanitize_filename(args.input_pdf.stem)}"]
                process_pdf(args.input_pdf, combined_lines=combined_lines)
                with open(combined_out, "w", encoding="utf-8") as f:
                    f.write("\n".join(combined_lines).rstrip() + "\n")
                print(f"Saved combined Markdown: {combined_out}")
            else:
                # Original behavior: write per-page files
                process_pdf(args.input_pdf, combined_lines=None)
    

if __name__ == "__main__":
    main()
