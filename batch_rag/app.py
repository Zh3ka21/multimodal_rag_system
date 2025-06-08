"""Streamlit app for performing multimodal (RAG) on 'The Batch' articles.

It combines FAISS-based text+image search with OpenAI GPT-based answer generation.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
import openai
import streamlit as st
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from batch_rag.logger import get_logger

# Load .env for OpenAI key
load_dotenv()

logger = get_logger("AppLogger")

@st.cache_resource
def load_resources() -> tuple[SentenceTransformer, SentenceTransformer, faiss.Index, list[dict]]:
    """Load required models and data for the RAG application.

    Returns:
        A tuple containing:
        - text_model: A sentence transformer for text embeddings.
        - image_model: A sentence transformer (CLIP) for image embeddings.
        - index: The FAISS index combining text + image vectors.
        - articles: A list of metadata dictionaries for articles.

    """
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    image_model = SentenceTransformer("clip-ViT-B-32")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    index = faiss.read_index(r"batch_data/multimodal_index.faiss")
    with Path.open(r"batch_data/metadata.json", "r", encoding="utf-8") as f:
        articles = json.load(f)
    return text_model, image_model, index, articles, clip_model, clip_processor

class RagApplication:
    """Multimodal Retrieval-Augmented Generation (RAG) application using Streamlit.

    Attributes:
        client (openai.OpenAI): OpenAI API client.
        text_model (SentenceTransformer): Model for text embeddings.
        image_model (SentenceTransformer): Model for image embeddings.
        index (faiss.Index): FAISS index for similarity search.
        articles (List[dict]): Metadata for indexed articles.

    """

    def __init__(self) -> None:
        """Initialize the RAG application and load resources."""
        st.set_page_config(page_title="Multimodal RAG", layout="wide")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.text_model, self.image_model,self.index, self.articles, self.clip_model, self.clip_processor = load_resources()

    def _generate_prompt(self, context: str, question: str) -> str:
        """Build the prompt for OpenAI completion using provided context and user question.

        Args:
            context (str): Textual context extracted from top articles.
            question (str): User query.

        Returns:
            str: Formatted prompt for OpenAI.

        """
        return f"""You are a helpful assistant. Use ONLY the context provided to answer.

        Context:
        {context}

        Question:
        {question}

        Only answer using information from the context above.
        If the answer is not present in the context, respond with:
        "Not found in context."
        """

    def query_openai(self, prompt: str) -> tuple[str, int, float]:
        """Make a query using openAI api.

        Args:
            prompt (str): user's prompt

        Returns:
            tuple[str, int, float]: answer from model, amount of tokens and cost spent.

        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": """
                            You are a helpful assistant that only uses the provided context.
                        """
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
                temperature=0.7,
                max_tokens=300,
            )
            answer = response.choices[0].message.content
            tokens = response.usage.total_tokens
            cost = tokens * 0.0015 / 1000
            logger.info(f"Prompt sent with {tokens} tokens, estimated cost: ${cost:.6f}")
            return answer, tokens, cost

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"OpenAI API error: {e}", 0, 0

    def _embed_query_as_image(self, query: str) -> np.ndarray:
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = self.clip_model.get_text_features(**inputs)[0].cpu().numpy()
        return emb / np.linalg.norm(emb)

    def run(self) -> None:
        """
        Launch the Streamlit interface and handle user interaction.
        Performs:
        - Query input
        - Multimodal retrieval (text + image)
        - Article preview
        - Context construction
        - OpenAI answer generation
        """
        st.title(" Multimodal RAG Search + Answer")
        st.markdown("""Search across **The Batch** articles using text + image
                    embeddings, and generate answers.""")

        query = st.text_input("Enter your query", "")

        if not query:
            st.info("Enter a query to get started.")
            return

        # Encode query
        text_emb = self.text_model.encode([query], convert_to_numpy=True)
        img_emb = self._embed_query_as_image(query).reshape(1, -1)
        query_vector = np.hstack((text_emb, img_emb)).astype("float32")

        # Search in FAISS index
        D, Ind = self.index.search(query_vector, k=3)

        st.subheader("Top Matching Articles")
        top_articles = []

        for i, idx in enumerate(Ind[0]):
            a = self.articles[idx]
            top_articles.append(a)
            st.markdown(f"### [{a['title']}]({a['url']})")
            if isinstance(a.get("image_url"), str) and a["image_url"].startswith("http"):
                st.image(a["image_url"], width=400)
                st.markdown(f"**Caption**: {a['image_caption']}")
            st.write(a["body"][:500] + "...")

        if not top_articles:
            st.warning("Sorry, we couldn't find a relevant article for your query.")
            return

        # Prepare context and query OpenAI
        context = "\n\n".join([
            f"Title: {a['title']}\n"
            f"Body: {a['body'][:500]}\n"
            f"Image: {a.get('image_caption', 'No image description')}"
            for a in top_articles
        ])        
        prompt = self._generate_prompt(context=context, question=query)

        with st.spinner("Generating answer..."):
            answer, tokens_used, cost = self.query_openai(prompt)

        st.subheader("Generated Answer")
        st.write(answer)

        logger.info(f"""
                    **Tokens used:** `{tokens_used}` &nbsp;&nbsp;
                    | &nbsp;&nbsp; **Estimated cost:** `${cost:.6f}`
                """)
