import json
import numpy as np #type: ignore
import faiss #type: ignore
from sentence_transformers import SentenceTransformer #type: ignore

# Load articles
with open("test_articles.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# Prepare embeddings using a lightweight transformer
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus = [f"{a['title']} {a['body']}" for a in articles]
embeddings = model.encode(corpus, convert_to_numpy=True).astype("float32")

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save outputs
faiss.write_index(index, "text_index.faiss")
np.save("embeddings.npy", embeddings)

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, indent=2)

print("✅ Index and metadata saved.")
