"""Browser smoke test against an isolated local app; never calls paid providers."""

import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import httpx
from img_creator.platform.db import Database, User
from img_creator.platform.security import password_hash

root = tempfile.TemporaryDirectory(prefix="img-ui-")
env = {
    **os.environ,
    "DATABASE_URL": "sqlite:///" + root.name + "/test.db",
    "PLATFORM_STORAGE_DIR": root.name + "/data",
    "COOKIE_SECURE": "false",
    "PUBLIC_URL": "http://127.0.0.1:8000",
    "BFL_API_KEY": "",
    "S3_BUCKET": "",
    "STRIPE_SECRET_KEY": "",
    "LOCAL_MODELS_ENABLED": "false",
    "IMG_CREATOR_SR_WEIGHTS": "",
}
db = Database(env["DATABASE_URL"])
db.initialize()
with db.transaction() as session:
    session.add(
        User(
            email="preview@example.com", password=password_hash("preview-only-long-password"), role="owner", credits=100
        )
    )
db.engine.dispose()
Path("test-results").mkdir(exist_ok=True)
server = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "img_creator.platform.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--no-proxy-headers",
    ],
    env=env,
)
try:
    with httpx.Client(trust_env=False) as client:
        for attempt in range(100):
            try:
                if client.get("http://127.0.0.1:8000/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("Preview did not start")
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = page.goto("http://127.0.0.1:8000/", wait_until="networkidle")
        assert response is not None
        policy = response.headers.get("content-security-policy", "")
        assert "script-src 'self'" in policy and "'unsafe-eval'" not in policy
        expect(page.locator('#upscale option[value="learned"]')).to_be_disabled()
        expect(page.locator("#download-native")).to_be_hidden()
        page.screenshot(path="test-results/studio-desktop.png", full_page=True)
        page.locator("#auth-open").click()
        page.locator("#email").fill("preview@example.com")
        page.locator("#password").fill("preview-only-long-password")
        page.locator("#auth-submit").click()
        page.locator('[data-page="admin"]').wait_for(state="visible")
        page.locator('[data-page="admin"]').click()
        page.locator("#users-table table").wait_for()
        page.screenshot(path="test-results/owner-desktop.png", full_page=True)
        page.locator('[data-admin="data"]').click()
        page.locator("#dataset-name").fill("Browser test collection")
        page.locator("#dataset-rights").fill("Original reference material for browser validation")
        page.locator("#dataset-form button").click()
        # Locator assertions retry without compiling a string inside the page.
        # Options are attached but not necessarily visible when a select is closed.
        expect(page.locator("#dataset-select option")).to_have_count(1)
        expect(page.locator("#dataset-select")).to_have_value(re.compile(r"^[0-9a-f]{32}$"))
        page.locator("#asset-files").set_input_files(
            {"name": "notes.txt", "mimeType": "text/plain", "buffer": b"Natural side lighting and realistic ceramics."}
        )
        page.locator("#upload-form button").click()
        page.locator("#assets-grid article").wait_for()
        assert "notes.txt" in page.locator("#assets-grid").inner_text()
        page.locator('[data-page="billing"]').click()
        page.locator("#plans .plan").first.wait_for()
        page.locator('[data-page="studio"]').click()
        page.locator("#model").select_option("flux-2-flex")
        page.locator("#effort").select_option("high")
        page.locator("#resolution").select_option("8K")
        page.locator("#prompt").fill("A handmade ceramic vase in natural window light")
        expect(page.locator("#quote")).to_contain_text("8192")
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path="test-results/studio-mobile.png", full_page=True)
        assert page.locator("html").evaluate("(root) => root.scrollWidth <= window.innerWidth")
        assert not errors, errors
        print(
            "Browser checks passed: owner login, dashboard, dataset creation/upload, billing view, model/effort/resolution quote, mobile overflow, no JS errors."
        )
        browser.close()
finally:
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    root.cleanup()
