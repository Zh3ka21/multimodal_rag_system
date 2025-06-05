import streamlit as st #type: ignore
import json
import faiss #type: ignore
import numpy as np #type: ignore
from sentence_transformers import SentenceTransformer #type: ignore

st.set_page_config(page_title="Multimodal RAG", layout="wide")

@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("text_index.faiss")
    with open("metadata.json", "r", encoding="utf-8") as f:
        articles = json.load(f)
    return model, index, articles

model, index, articles = load_resources()

st.title("📰 Multimodal RAG News Search")
st.markdown("Search across **The Batch** articles with text + image results.")

query = st.text_input("Enter your query", "open source AI models")

if query:
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    D, Ind = index.search(query_embedding, k=5)

    st.subheader("Top Matching Articles:")

    for idx in Ind[0]:
        a = articles[idx]
        st.markdown(f"### [{a['title']}]({a['url']})")
        if a.get("image_url", "").startswith("http"):
            st.image(a["image_url"], width=400)
        st.write(a["body"][:500] + "...")
