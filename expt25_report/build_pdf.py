import markdown, base64, re, os
from weasyprint import HTML
md=open("expt25_paper.md").read()
html_body=markdown.markdown(md, extensions=["tables","fenced_code","attr_list"])
def inline(m):
    alt,src=m.group(1),m.group(2)
    if not os.path.exists(src): return m.group(0)
    b64=base64.b64encode(open(src,"rb").read()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}"/>'
html_body=re.sub(r'<img alt="([^"]*)" src="([^"]+)"\s*/?>',inline,html_body)
css="""
@page { size: A4; margin: 1.7cm 1.6cm;
  @bottom-center { content: "MoE Breakdown — Transport Ablation (expt2+expt2.5)   ·   " counter(page) " / " counter(pages);
    font-size: 8pt; color: #999; } }
* { box-sizing: border-box; }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.6pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 19pt; color: #14213d; border-bottom: 3px solid #2a9d8f; padding-bottom: 7px; margin-bottom: 3px; line-height: 1.25; }
h2 { font-size: 13.5pt; color: #14213d; margin-top: 20px; border-bottom: 1px solid #dde; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 11pt; color: #2a6f97; margin-top: 14px; page-break-after: avoid; }
p { margin: 6px 0; text-align: justify; }
strong { color: #0b2545; }
em { color: #333; }
a { color: #2a6f97; text-decoration: none; }
code { background: #eef1f4; padding: 1px 4px; border-radius: 3px; font-family: 'DejaVu Sans Mono', monospace; font-size: 8.2pt; color: #b5532a; }
table { border-collapse: collapse; width: 100%; margin: 11px 0; font-size: 8.4pt; page-break-inside: avoid; }
th { background: #14213d; color: #fff; padding: 5px 7px; text-align: left; }
td { border: 1px solid #d6dbe0; padding: 4px 7px; }
tr:nth-child(even) td { background: #f6f8fa; }
img { max-width: 100%; height: auto; display: block; margin: 9px auto; border: 1px solid #e3e6ea; border-radius: 4px; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #e3e6ea; margin: 16px 0; }
ul, ol { margin: 6px 0 6px 2px; padding-left: 20px; }
li { margin: 3px 0; }
/* Abstract emphasis */
h2:first-of-type + p { background:#f7fbfa; border-left:4px solid #2a9d8f; padding:8px 12px; font-size:9.4pt; }
"""
full=f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html_body}</body></html>"
HTML(string=full,base_url=".").write_pdf("expt25_paper.pdf")
print("PDF:", os.path.getsize("expt25_paper.pdf"),"bytes")
