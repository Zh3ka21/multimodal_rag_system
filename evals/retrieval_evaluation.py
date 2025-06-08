import json
import faiss
import numpy as np
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import torch
from transformers import CLIPModel, CLIPProcessor

# -------------------------------
# Config: file paths
# -------------------------------
INDEX_PATH = "batch_data/multimodal_index.faiss"
METADATA_PATH = "batch_data/metadata.json"
EVAL_DATASET_PATH = "evals/eval_dataset.json"

# -------------------------------
# Load Models and Data
# -------------------------------
print("🔧 Loading models and index...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = SentenceTransformer("all-MiniLM-L6-v2")


index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
    eval_dataset = json.load(f)

# -------------------------------
# Evaluation Function
# -------------------------------
def embed_query_as_image_text(query: str) -> np.ndarray:
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)[0].cpu().numpy()
    return emb / np.linalg.norm(emb)

def fuzzy_match(title: str, expected_titles: List[str]) -> bool:
    return any(exp.lower() in title.lower() or title.lower() in exp.lower() for exp in expected_titles)

def evaluate_query(query: str, expected_titles: List[str], top_k: int = 3) -> Dict:
    """Embed query, search FAISS, compare with expected titles using fuzzy matching."""
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    img_emb = embed_query_as_image_text(query).reshape(1, -1) 
    full_query_vector = np.hstack((query_emb, img_emb)).astype("float32")

    D, Index = index.search(full_query_vector, k=top_k)
    top_indices = Index[0]
    top_titles = [metadata[i]["title"] for i in top_indices]

    top1_match = int(fuzzy_match(top_titles[0], expected_titles))
    topk_recall = sum([1 for t in top_titles if fuzzy_match(t, expected_titles)]) / len(expected_titles)

    return {
        "query": query,
        "expected": expected_titles,
        "retrieved": top_titles,
        "top1_match": top1_match,
        "topk_recall": round(topk_recall, 2)
    }


# -------------------------------
# Run Evaluation
# -------------------------------
print("📊 Running evaluation...\n")
results = [evaluate_query(item["query"], item["expected_titles"], top_k=3) for item in eval_dataset]

# -------------------------------
# Print Summary
# -------------------------------
total = len(results)
top1_hits = sum(r["top1_match"] for r in results)
avg_recall = sum(r["topk_recall"] for r in results) / total

print("========= Evaluation Summary =========")
print(f"🔎 Total queries evaluated : {total}")
print(f"✅ Top-1 Accuracy          : {top1_hits / total:.2f}")
print(f"📈 Avg. Top-3 Recall       : {avg_recall:.2f}")
print("======================================\n")

for r in results:
    print(f"📝 Query: {r['query']}")
    print(f"  - Expected Titles : {r['expected']}")
    print(f"  - Retrieved Titles: {r['retrieved']}")
    print(f"  - Top-1 Match     : {r['top1_match']} | Top-3 Recall: {r['topk_recall']}")
    print("-" * 60)
