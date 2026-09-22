"""Step 12 (optional) - screenshots of the running app for the report / presentation.

Needs `pip install playwright` and an installed Edge or Chrome. The script starts the app itself on port 8599.
Run:  python scripts/12_app_screenshots.py
Output: outputs/figures/app/*.png
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "app"; OUT.mkdir(parents=True, exist_ok=True)
PORT = 8599
server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless", "true", "--server.port", str(PORT),
                           "--browser.gatherUsageStats", "false"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            if requests.get(f"http://localhost:{PORT}/_stcore/health", timeout=2).text == "ok":
                break
        except requests.RequestException:
            time.sleep(1)
    with sync_playwright() as p:
        browser = None
        for channel in ("msedge", "chrome"):
            try:
                browser = p.chromium.launch(channel=channel, headless=True); break
            except Exception:
                continue
        page = browser.new_page(viewport={"width": 1500, "height": 3400})
        page.goto(f"http://localhost:{PORT}", wait_until="networkidle", timeout=120000)
        page.get_by_text("Quick example").first.wait_for(timeout=120000)

        def choose(label_regex, option_regex):
            page.get_by_role("combobox", name=re.compile(label_regex)).click()
            page.keyboard.type(option_regex.lstrip("^").split(":")[0])          # the option list is virtualised: filter it first
            page.get_by_role("option", name=re.compile(option_regex)).first.click()
            page.wait_for_timeout(2500)

        def report_and_shot(name):
            page.get_by_role("button", name=re.compile("Generate report")).click()
            page.get_by_text(re.compile("Faithfulness validation")).first.wait_for(timeout=120000)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / name), full_page=True)

        choose("Quick example", r"^13 DoS: detected")
        report_and_shot("app_1_detect_dos.png")
        choose("Quick example", r"^17 Data replay Sybil: missed")
        report_and_shot("app_2_detect_missed.png")
        for tab, name in (("Model performance", "app_3_performance.png"), ("Global explainability", "app_4_global_explainability.png"), ("About & limitations", "app_5_about.png")):
            page.get_by_role("tab", name=re.compile(tab)).click(); page.wait_for_timeout(4000)
            page.screenshot(path=str(OUT / name), full_page=True)
        browser.close()
    print("screenshots written to", OUT)
finally:
    server.terminate()
