import json, sys
from playwright.sync_api import sync_playwright

URL = "http://localhost:4173/"

with sync_playwright() as p:
    browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        args=["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal"],
    )
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(URL, wait_until="domcontentloaded")
    info = page.evaluate("""async () => {
        const out = {hasGpu: !!navigator.gpu, adapter: null, err: null};
        if (navigator.gpu) {
          try { const a = await navigator.gpu.requestAdapter(); out.adapter = a ? (a.info?.vendor + ' / ' + a.info?.architecture) : null; }
          catch (e) { out.err = String(e); }
        }
        return out;
    }""")
    print("webgpu probe:", json.dumps(info))
    print("title:", page.title())
    browser.close()
