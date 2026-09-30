"""
Render SUMMARY.md and EVALUATION_REPORT.md into polished, self-contained,
print-ready HTML (open in a browser; Cmd/Ctrl+P -> Save as PDF to hand out).

    python eval/build_report_html.py
"""
from pathlib import Path
import markdown

BASE = Path(__file__).resolve().parent.parent

CSS = """
:root { --ink:#1a2233; --muted:#5b6472; --line:#e3e8ef; --accent:#2f5fe0;
        --accent-soft:#eef3ff; --ok:#0f7b3f; --warn:#b26a00; --code:#0b1f4d; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       color: var(--ink); line-height: 1.55; max-width: 880px; margin: 0 auto;
       padding: 56px 48px 80px; font-size: 15px; background:#fff; }
h1 { font-size: 30px; letter-spacing:-.5px; margin: 0 0 4px; padding-bottom: 14px;
     border-bottom: 3px solid var(--accent); }
h2 { font-size: 21px; margin: 38px 0 12px; padding-top: 10px; color: var(--ink);
     border-top: 1px solid var(--line); padding-top: 22px; }
h3 { font-size: 16.5px; margin: 26px 0 8px; color: var(--accent); }
h4 { font-size: 14px; margin: 18px 0 6px; color: var(--muted); text-transform: uppercase;
     letter-spacing:.4px; }
p, li { color: var(--ink); }
a { color: var(--accent); text-decoration: none; }
strong { color: var(--ink); }
code { background:#f4f6fa; color: var(--code); padding: 1.5px 5px; border-radius: 4px;
       font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace; font-size: 13px; }
pre { background:#0f1729; color:#e6ebff; padding: 14px 16px; border-radius: 8px;
      overflow-x:auto; font-size: 12.5px; line-height:1.5; }
pre code { background: none; color: inherit; padding: 0; }
blockquote { margin: 12px 0; padding: 10px 16px; background: var(--accent-soft);
             border-left: 4px solid var(--accent); border-radius: 0 6px 6px 0; color: var(--ink); }
blockquote p { margin: 4px 0; }
table { border-collapse: collapse; width: 100%; margin: 14px 0 20px; font-size: 13.5px;
        box-shadow: 0 1px 0 var(--line); }
th { background: var(--ink); color:#fff; text-align: left; padding: 9px 12px; font-weight: 600; }
td { padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
tr:nth-child(even) td { background:#f8fafc; }
hr { border:none; border-top: 1px solid var(--line); margin: 30px 0; }
em { color: var(--muted); }
.doc-meta { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
@media print {
  body { padding: 0; font-size: 11.5pt; max-width: none; }
  h2 { page-break-after: avoid; } table, pre, blockquote { page-break-inside: avoid; }
  a { color: var(--ink); }
}
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{css}</style></head>
<body>{body}
<hr><p class="doc-meta">Enterprise Knowledge Assistant · generated {ts} · print to PDF via your browser (Cmd/Ctrl+P).</p>
</body></html>"""


def render(md_path: Path, out_path: Path, title: str, ts: str) -> None:
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"])
    body = md.convert(md_path.read_text())
    out_path.write_text(TEMPLATE.format(title=title, css=CSS, body=body, ts=ts))
    print(f"  wrote {out_path.relative_to(BASE)}")


if __name__ == "__main__":
    import datetime
    ts = datetime.date.today().isoformat()
    print("Rendering presentation HTML:")
    render(BASE / "SUMMARY.md", BASE / "SUMMARY.html",
           "Evaluation Summary — Enterprise Knowledge Assistant", ts)
    render(BASE / "EVALUATION_REPORT.md", BASE / "EVALUATION_REPORT.html",
           "Evaluation Report — Enterprise Knowledge Assistant", ts)
    print("Done. Open the .html files in a browser and Save as PDF to hand out.")
