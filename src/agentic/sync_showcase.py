"""Copy SHOWCASE.html (the 15D/5% model showcase, rebuilt weekly by build_recreation_kit.py) to the website's
"Older Models" tab: shreeganeshmodel-deploy/older-models/index.html.

Since 2026-10-16 the site's home page (shreeganeshmodel-deploy/index.html) is SGM Data World, so the showcase no longer
overwrites it. The copy gets root-relative asset paths (it now lives one folder down) and a small top-right link back to
the home page. Idempotent; called from run_weekly_pipeline.sh step 4.5.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC, DEST = ROOT / "SHOWCASE.html", ROOT / "shreeganeshmodel-deploy/older-models/index.html"
NAV = ('<a href="/" style="position:fixed;top:14px;right:16px;z-index:1000;padding:7px 14px;border-radius:999px;'
       'background:#14141d;color:#e8e8ee;border:1px solid #2a2a3a;font:600 13px -apple-system,system-ui,sans-serif;'
       'text-decoration:none">SGM Data World</a>')


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    html = SRC.read_text()
    html = re.sub(r'(src|href)="assets/', r'\1="/assets/', html)
    html, n = re.subn(r"(<body[^>]*>)", r"\1\n" + NAV, html, count=1)
    if n != 1:
        print("no <body> tag found", file=sys.stderr)
        return 1
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(html)
    print(f"wrote {DEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
