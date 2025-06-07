# batch_processor.py
import os
import json
import faiss
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from urllib.parse import urljoin, urlparse, parse_qs
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer
import torch
from transformers import CLIPModel, CLIPProcessor

from batch_rag.logger import get_logger

logger = get_logger("BatchProcessor")

class BatchProcessor:
    BASE_URL = "https://www.deeplearning.ai"
    START_URL = "https://www.deeplearning.ai/the-batch"

    def __init__(self, output_dir="batch_data"):
        self.output_dir = output_dir
        self.article_path = os.path.join(output_dir, "articles.json")
        self.index_path = os.path.join(output_dir, "multimodal_index.faiss")
        self.metadata_path = os.path.join(output_dir, "metadata.json")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("backups", exist_ok=True)

        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _caption_image_clip(self, image: Image.Image) -> str:
        """
        Generates a simple caption for an image using CLIP zero-shot classification.
        """

        candidate_captions = [
            "Robots building house",
            "Whale with microcircuits(Deepseek)",
            "Bar chart Benchmarks Accuracy",
            "Male and female talking in front of monitor",
            "Research bar chart reversed",
            "Female and a male sitting in sandbox",
            "Accuracy table of different models",
            "Speedometer",
            "Chart with people photos",
            "Green text on black background in C","Two people talking seating at computers",
            "Sneakers",
            "Female and a male on chaise lounge","","4 Photo with a lot of people",
            "Male seating at a computer and female showing thumbs up",
            "Pipeline Flowchart",
            "Pipeline Flowchart Voice stack",
            "White thermal vision detected person",            
        ]

        inputs = self.image_processor(text=candidate_captions, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.image_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        best_idx = probs[0].argmax().item()
        return candidate_captions[best_idx]

    def _extract_real_image_url(self, src):
        """
        Cleans and resolves the image src to a usable direct URL.
        """
        if not src:
            return None

        if src.startswith("data:"):
            return None  # Base64 placeholder image

        parsed = urlparse(src)
        if "_next/image" in parsed.path:
            qs = parse_qs(parsed.query)
            real_url = qs.get("url", [None])[0]
            if real_url:
                return urljoin(self.BASE_URL, real_url) if real_url.startswith("/") else real_url
            return None

        # If already a valid URL, return as-is
        if parsed.scheme in ["http", "https"]:
            return src

        # Otherwise treat as relative and resolve
        return urljoin(self.BASE_URL, src)

    def extract_articles(self):
        logger.info("Starting article extraction")       
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(self.START_URL)
            page.wait_for_selector("article")

            articles_meta = []
            articles = page.query_selector_all("article")
            for idx, article in enumerate(articles):
                try:
                    title_elem = article.query_selector("h2, h3")
                    title = title_elem.inner_text().strip() if title_elem else "No title"

                    date_elem = article.query_selector("a > div")
                    date = date_elem.inner_text().strip() if date_elem else "Unknown Date"

                    img_elem = article.query_selector("img")
                    raw_src = img_elem.get_attribute("src") if img_elem else None
                    image_url = self._extract_real_image_url(raw_src)

                    link_elem = article.query_selector("a[href]")
                    href = link_elem.get_attribute("href") if link_elem else None
                    article_url = urljoin(self.BASE_URL, href) if href else None

                    articles_meta.append({
                        "title": title,
                        "date": date,
                        "url": article_url,
                        "image_url": image_url,
                    })
                except Exception as e:
                    logger.warning(f"[Metadata Error] Article #{idx}: {e}")

            for idx, item in enumerate(articles_meta):
                logger.info(f"[{idx + 1}/{len(articles_meta)}] Scraping: {item['title']}")
                if not item["url"]:
                    item["body"] = ""
                    continue
                try:
                    page.goto(item["url"], timeout=15000)
                    page.wait_for_selector("article", timeout=5000)
                    body_elem = page.query_selector("article")
                    raw_body = body_elem.inner_text().strip() if body_elem else ""
                    item["body"] = self._truncate_body(raw_body)
                except Exception as e:
                    logger.warning(f"[Body Error] Failed to load {item['url']}: {e}")
                    item["body"] = ""

            with open(self.article_path, "w", encoding="utf-8") as f:
                json.dump(articles_meta, f, indent=2, ensure_ascii=False)

            self._backup_articles()

            browser.close()
        logger.info(f"✅ Articles saved to {self.article_path}")

    def _backup_articles(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_path = Path("backups") / f"articles_{timestamp}.json"
        try:
            with open(self.article_path, "r", encoding="utf-8") as src, open(backup_path, "w", encoding="utf-8") as dest:
                dest.write(src.read())
            logger.info(f"📦 Backup created: {backup_path}")
        except Exception as e:
            logger.error(f"❌ Failed to backup articles: {e}")

    def _truncate_body(self, text: str, max_tokens=500) -> str:
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        sentences = text.split(". ")
        selected = []
        token_count = 0
        for sent in sentences:
            tokens = tokenizer.encode(sent)
            if token_count + len(tokens) > max_tokens:
                break
            selected.append(sent)
            token_count += len(tokens)
        return ". ".join(selected)

    def build_index(self):
        logger.info("Building multimodal index")

        if not os.path.exists(self.article_path):
            raise FileNotFoundError("Run extract_articles() first")

        with open(self.article_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        text_data = []
        image_embeddings = []

        for idx, a in enumerate(articles):
            text_data.append(f"{a['title']} {a['body']}")
            try:
                url = a.get("image_url", "")
                is_real_image = url.startswith("http") and not url.lower().endswith(
                    ("placeholder.png", "default.jpg", "image.png")
                )
                if is_real_image:
                    response = requests.get(url, timeout=5)
                    img = Image.open(BytesIO(response.content)).convert("RGB")

                    # Caption the image and store it
                    caption = self._caption_image_clip(img)
                    a["image_caption"] = caption

                    inputs = self.image_processor(images=img, return_tensors="pt")
                    with torch.no_grad():
                        img_emb = self.image_model.get_image_features(**inputs)
                        img_emb = img_emb[0].cpu().numpy()
                        img_emb = img_emb / np.linalg.norm(img_emb)
                else:
                    raise ValueError("Dummy or invalid image")
            except Exception as e:
                logger.warning(f"[Image Error] Article #{idx} - using dummy image: {e}")
                img_emb = np.zeros(512, dtype="float32")
                a["image_caption"] = ""
            image_embeddings.append(img_emb)

        text_embeddings = self.text_model.encode(text_data, convert_to_numpy=True).astype("float32")
        image_embeddings = np.array(image_embeddings).astype("float32")

        combined = np.hstack((text_embeddings, image_embeddings))
        index = faiss.IndexFlatL2(combined.shape[1])
        index.add(combined)

        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2)

        logger.info("Index + metadata saved.")

    def run_all(self):
        self.extract_articles()
        self.build_index()
