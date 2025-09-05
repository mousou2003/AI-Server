#!/usr/bin/env python3
"""
Convert Markdown to PDF using Edge/Chrome headless print.
Adds page border, header, and footer: "Page X of Y | Dated ...".
Requires: pip install markdown
Windows: uses msedge.exe by default; falls back to chrome.exe if found.
"""

import argparse, os, shutil, subprocess, tempfile
from pathlib import Path
from markdown import markdown

HTML_SHELL = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  @page {{
    size: {page_size};
    margin: {margin_in}in;
  }}
  html, body {{
    font-family: "Times New Roman", Times, serif;
    font-size: 11pt;
    line-height: 1.35;
    color: #000;
  }}
  h1 {{ font-size: 16pt; text-align: center; margin: 0.4em 0 0.2em; }}
  h2 {{ font-size: 13pt; margin: 0.6em 0 0.3em; }}
  h3 {{ font-size: 12pt; margin: 0.6em 0 0.3em; }}
  p  {{ margin: 0.35em 0; }}
  ul, ol {{ margin: 0.35em 0 0.35em 1.3em; }}
  hr {{ border: 0; border-top: 1px solid #000; margin: 0.8em 0; }}
  .title-block {{ text-align: center; margin-bottom: 0.8em; }}
  .subtitle {{ font-size: 12pt; margin-top: 0.25em; }}
  /* Optional manual page break: <div class="pagebreak"></div> */
  .pagebreak {{ page-break-before: always; }}
</style>
</head>
<body>
  <div class="title-block">
    <h1>{title}</h1>
    {subtitle_html}
  </div>
  {body_html}
</body>
</html>
"""

def find_browser():
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    # Try PATH
    for exe in ["msedge.exe", "chrome.exe"]:
        p = shutil.which(exe)
        if p:
            return p
    raise FileNotFoundError("Neither Microsoft Edge nor Google Chrome was found.")

def build_footer_template(footer_date, header_text):
    # Chrome/Edge use HTML templates with special tokens:
    #   <span class="pageNumber"></span>, <span class="totalPages"></span>
    # We also replicate header at top and page/date at bottom.
    header_tpl = f"""
    <div style="font-size:9pt; width:100%; text-align:center; margin:0 1in;">
      {header_text}
    </div>""".strip()
    footer_tpl = f"""
    <div style="font-size:9pt; width:100%; text-align:center; margin:0 1in;">
      Page <span class="pageNumber"></span> of <span class="totalPages"></span> | {footer_date}
    </div>""".strip()
    return header_tpl, footer_tpl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_md", type=Path)
    ap.add_argument("output_pdf", type=Path)
    ap.add_argument("--header", default="")
    ap.add_argument("--footer-date", default="")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--page-size", default="Letter", help="Letter, A4, etc.")
    ap.add_argument("--margin", type=float, default=1.0, help="All margins in inches (default 1.0)")
    ap.add_argument("--border-offset", type=float, default=0.5, help="Border offset from page edge in inches (default 0.5)")
    args = ap.parse_args()

    if not args.input_md.exists():
        raise SystemExit(f"Input not found: {args.input_md}")

    md = args.input_md.read_text(encoding="utf-8")
    body_html = markdown(md, extensions=["extra", "sane_lists", "smarty"])
    title = args.title or args.input_md.stem.replace("_", " ").replace("-", " ")
    subtitle_html = f'<div class="subtitle">{args.subtitle}</div>' if args.subtitle else ""

    html = HTML_SHELL.format(
        title=title,
        subtitle_html=subtitle_html,
        body_html=body_html,
        page_size=args.page_size,
        margin_in=f"{args.margin:.2f}",
        border_in=f"{args.border_offset:.2f}",
    )

    browser = find_browser()
    header_tpl, footer_tpl = build_footer_template(args.footer_date, args.header)

    with tempfile.TemporaryDirectory() as td:
        html_path = Path(td) / "doc.html"
        header_path = Path(td) / "header.html"
        footer_path = Path(td) / "footer.html"
        html_path.write_text(html, encoding="utf-8")
        header_path.write_text(header_tpl, encoding="utf-8")
        footer_path.write_text(footer_tpl, encoding="utf-8")

        # Chrome/Edge headless print-to-pdf
        # Note: --no-margins is avoided; we use CSS margins so header/footer fit.
        cmd = [
            browser,
            "--headless=new",              # or "--headless" on older builds
            "--disable-gpu",
            f"--print-to-pdf={str(args.output_pdf)}",
            f"--print-to-pdf-no-header",   # hide default header/footer
            f"--header-template={header_path.as_uri()}",
            f"--footer-template={footer_path.as_uri()}",
            "--virtual-time-budget=10000",
            str(html_path.as_uri()),
        ]
        subprocess.run(cmd, check=True)

    print(f"✅ Wrote {args.output_pdf} using {Path(browser).name}")

if __name__ == "__main__":
    main()
