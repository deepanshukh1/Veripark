"""Screenshot the HTML report for the README (needs: pip install playwright && playwright install chromium)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

report = Path(sys.argv[1] if len(sys.argv) > 1 else "results/demo/report.html").resolve()
out = Path("docs/report_screenshot.png")
out.parent.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1100, "height": 900})
    page.goto(report.as_uri())
    page.screenshot(path=str(out), full_page=True)
    browser.close()
print(f"wrote {out}")
