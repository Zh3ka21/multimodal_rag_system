from playwright.sync_api import sync_playwright
import os
import json
from urllib.parse import urljoin

BASE_URL = "https://www.deeplearning.ai"
START_URL = "https://www.deeplearning.ai/the-batch"
OUTPUT_FOLDER = "batch_articles"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_articles():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(START_URL)
        page.wait_for_selector("article")

        article_meta = []

        # Pass 1: Collect all article metadata without leaving the page
        articles = page.query_selector_all("article")
        for idx, article in enumerate(articles):
            try:
                title_elem = article.query_selector("h2, h3")
                title = title_elem.inner_text().strip() if title_elem else "No title"

                date_elem = article.query_selector("a > div")
                date = date_elem.inner_text().strip() if date_elem else "Unknown Date"

                img_elem = article.query_selector("img")
                raw_src = img_elem.get_attribute("src") if img_elem else None
                image_url = urljoin(BASE_URL, raw_src) if raw_src else None

                link_elem = article.query_selector("a[href]")
                href = link_elem.get_attribute("href") if link_elem else None
                article_url = urljoin(BASE_URL, href) if href else None

                article_meta.append({
                    "title": title,
                    "date": date,
                    "url": article_url,
                    "image_url": image_url,
                })

            except Exception as e:
                print(f"[Metadata Error] Article #{idx}: {e}")

        # Pass 2: Visit each article URL to get body content
        for idx, item in enumerate(article_meta):
            print(f"[{idx+1}/{len(article_meta)}] Scraping article: {item['title']}")
            if not item["url"]:
                item["body"] = ""
                continue
            try:
                page.goto(item["url"], timeout=15000)
                page.wait_for_selector("article", timeout=5000)
                body_elem = page.query_selector("article")
                item["body"] = body_elem.inner_text().strip() if body_elem else ""
            except Exception as e:
                print(f"[Body Error] Failed to load {item['url']}: {e}")
                item["body"] = ""

        # Save to JSON
        with open(os.path.join(OUTPUT_FOLDER, "articles.json"), "w", encoding="utf-8") as f:
            json.dump(article_meta, f, indent=2, ensure_ascii=False)

        browser.close()


extract_articles()
