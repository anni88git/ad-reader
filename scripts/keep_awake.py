"""Open the hosted app in a real (headless) browser so Streamlit Community Cloud counts a visit; if the app is
asleep, press its wake-up button and wait until the app answers. Used by .github/workflows/keep-awake.yml.

  python scripts/keep_awake.py [URL]
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://ad-reader-kejd8ymzrybmufjazbakbc.streamlit.app/"
WAKE_TEXT = "get this app back up"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=os.getenv("BROWSER_CHANNEL") or None)   # e.g. "chrome" locally
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
        time.sleep(10)
        wake = page.get_by_role("button", name=WAKE_TEXT, exact=False)
        if wake.count():
            print("app was asleep - waking it up")
            wake.first.click()
            time.sleep(150)                     # restart + dependency install + model warm-up
        else:
            print("app is awake")
        # the app itself lives in an iframe on *.streamlit.app; wait for any Streamlit content to render
        for _ in range(30):
            if any("stApp" in (f.content() or "") for f in page.frames if f.url.startswith("https://")):
                print("app responded")
                break
            time.sleep(5)
        else:
            print("app did not render within the wait time (it may still be starting)")
        time.sleep(20)                          # stay connected briefly so the visit is counted
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
