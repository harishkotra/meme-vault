"""End-to-end test: real model, real images, real search, then offline search."""
import glob
import json
import os
import sys
import time
from playwright.sync_api import sync_playwright

URL = "http://localhost:4173/"
HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = sorted(
    p for p in glob.glob(os.path.join(HERE, "..", "test-images", "*")) if p.lower().endswith((".jpg", ".png"))
)

errors: list[str] = []
ok = True


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


with sync_playwright() as p:
    browser = p.chromium.launch(
        channel="chrome",
        headless=True,
        args=["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--use-angle=metal"],
    )
    context = browser.new_context(viewport={"width": 1500, "height": 950}, device_scale_factor=2)
    page = context.new_page()
    page.on("console", lambda msg: errors.append(f"console.{msg.type}: {msg.text}") if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))

    log(f"{len(IMAGES)} images: {[os.path.basename(i) for i in IMAGES]}")
    page.goto(URL, wait_until="domcontentloaded")

    # 1. wait for the model to be ready
    started = time.time()
    page.wait_for_function("() => document.querySelector('.app')?.dataset.phase === 'ready'", timeout=600_000)
    device = page.evaluate("() => document.querySelector('.app')?.dataset.device")
    log(f"model ready in {time.time() - started:.1f}s on {device}")
    page.screenshot(path="shots/01-empty.png")

    # 2. add the images
    started = time.time()
    page.locator("input[type=file]").first.set_input_files(IMAGES)
    page.wait_for_function(
        f"() => document.querySelector('.app')?.dataset.indexed === '{len(IMAGES)}'", timeout=600_000
    )
    log(f"indexed {len(IMAGES)} images in {time.time() - started:.1f}s")
    page.wait_for_timeout(300)
    page.screenshot(path="shots/02-vault.png")

    def search(text):
        before = int(page.evaluate("() => document.querySelector('.app')?.dataset.searchSeq ?? '0'"))
        page.fill(".searchbar input", text)
        page.wait_for_function(
            f"() => Number(document.querySelector('.app')?.dataset.searchSeq ?? 0) > {before}",
            timeout=180_000,
        )
        return page.evaluate(
            """() => Array.from(document.querySelectorAll('.tile')).map(t => ({
                 name: t.querySelector('img')?.alt,
                 score: t.querySelector('.score')?.textContent ?? null,
               }))"""
        )

    results = {}
    for query in ["a cat", "pizza", "the eiffel tower", "a banana", "a painting of a woman", "a golden retriever"]:
        hits = search(query)
        results[query] = hits
        top = hits[0] if hits else None
        log(f"{query!r} -> {top['name'] if top else 'nothing'} ({top['score'] if top else '-'}), {len(hits)} hits")

    page.screenshot(path="shots/03-search.png")

    # 3. the offline claim, tested rather than asserted
    context.set_offline(True)
    page.wait_for_timeout(400)
    offline_hits = search("a cat")
    offline_badge = page.evaluate("() => document.querySelector('.pill-live')?.textContent ?? null")
    log(f"OFFLINE search 'a cat' -> {offline_hits[0]['name'] if offline_hits else 'nothing'} ({offline_hits[0]['score'] if offline_hits else '-'})")
    log(f"offline badge: {offline_badge!r}")
    page.screenshot(path="shots/04-offline.png")
    context.set_offline(False)
    page.wait_for_timeout(200)

    # 4. lightbox
    page.click(".tile")
    page.wait_for_timeout(900)
    page.screenshot(path="shots/05-lightbox.png")
    lightbox = page.evaluate("() => !!document.querySelector('.figure img')")
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    closed = page.evaluate("() => !document.querySelector('.figure')")
    log(f"lightbox opened={lightbox} closed_by_escape={closed}")

    # 5. attribution renders with the right hrefs
    credit = page.evaluate(
        """() => Array.from(document.querySelectorAll('.credit a')).map(a => [a.textContent.trim(), a.href])"""
    )
    log(f"credit links: {credit}")

    # 6. persistence across a reload
    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("() => document.querySelector('.app')?.dataset.phase === 'ready'", timeout=600_000)
    total = page.evaluate("() => document.querySelector('.app')?.dataset.total")
    reopened = search("pizza")
    log(f"after reload: {total} memes kept, 'pizza' -> {reopened[0]['name'] if reopened else 'nothing'}")
    page.screenshot(path="shots/06-reloaded.png")

    print("\nRESULTS", json.dumps(results, indent=1)[:2500])
    print("ERRORS", json.dumps(errors[:20], indent=1))
    browser.close()

# top-1 must be the right thing; the text-overlay memes made from these photos are also right
expect = {
    "a cat": {"cat.jpg"},
    "pizza": {"pizza.jpg"},
    "the eiffel tower": {"eiffel.jpg"},
    "a banana": {"banana.jpg", "meme-banana.png"},
    "a painting of a woman": {"monalisa.jpg"},
    "a golden retriever": {"dog.jpg", "meme-dog.png"},
}
failed = [q for q, want in expect.items() if not results.get(q) or results[q][0]["name"] not in want]
for q, want in expect.items():
    rows = results.get(q) or []
    if len(rows) > 1:
        print(f"  gap {q!r}: {rows[0]['score']} vs {rows[1]['score']}")
checks = {
    "rank1": not failed,
    "offline_top": bool(offline_hits) and offline_hits[0]["name"] == "cat.jpg",
    "persistence": total == str(len(IMAGES)) and bool(reopened) and reopened[0]["name"] == "pizza.jpg",
    "lightbox": lightbox and closed,
    "no_console_errors": not errors,
    "credit": credit == [
        ["Built By Harish Kotra", "https://harishkotra.me/"],
        ["Checkout my other builds", "https://dailybuild.xyz/"],
    ],
}
print("\nCHECKS", json.dumps(checks))
if failed:
    print("RANK-1 FAILURES", failed)
sys.exit(0 if all(checks.values()) else 1)