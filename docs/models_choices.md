# Model Selection and Justification

This document outlines the models chosen for the multimodal RAG system, and explains the reasoning behind each decision, taking into account accuracy, cost, and hardware limitations.

---

## Language Model for Answer Generation

### Chosen Model: OpenAI `gpt-3.5-turbo` (16K context)
I initially tested open-source alternatives, such as `TinyLlama-1.1B` from Hugging Face.

### DRAWBACKS: 
Although lightweight, it **struggled with context adherence** — often generating hallucinated content or ignoring the provided context. These models failed to reliably follow the instruction to **“ONLY use the given context,”** which is essential for RAG-style generation.

### GPT-3.5-Turbo?

ADVANTAGES:
Offers **strong instruction-following**, ideal for constrained-answer generation.
16K token context is large enough to support multiple articles and long questions.
Very cost-effective:
  - Input: **$0.0015 / 1K tokens**
  - Output: **$0.002 / 1K tokens**
Easy to integrate and **requires no local GPU or fine-tuning**, making it ideal for a limited-resource environment such as mine.

---

## Embedding Models for Retrieval

### Text Embedding: `all-MiniLM-L6-v2`
Small, fast, and efficient for both training and inference.
Performs well for semantic similarity while maintaining low memory usage.
Ideal for **CPU-only environments**.
Free of charge.

### Image Embedding: `openai/clip-vit-base-patch32`
Powerful **CLIP model** supporting aligned text/image embeddings.
Used to encode:
  - Article images
  - Text queries (as if they were image captions) to simulate visual intent
Runs well on **CPU**, with no GPU requirement for inference.
Free of charge.

---

## Prioritization Criteria

When selecting models, I prioritized:
1. **Instructional reliability** (answers grounded in context)
2. **Multimodal support** (joint text + image representation)
3. **CPU & memory efficiency** (no GPU available)
4. **Cost-effective inference**
5. **Ease of access via OpenAI or Hugging Face**

---

## Model Summary Table

| Task                  | Model Used                      | Reasoning                                     |
| --------------------- | ------------------------------- | --------------------------------------------- |
| Answer Generation     | GPT-3.5-Turbo (OpenAI)          | Accurate, cheap, 16K context, no GPU required |
| Text Embedding        | all-MiniLM-L6-v2                | Small, fast, CPU-friendly                     |
| Image Embedding       | openai/clip-vit-base-patch32    | Multimodal, works with CLIP, runs on CPU      |
| Query Image Embedding | `CLIP.get_text_features(query)` | Enables text queries to simulate image input  |

---

> This combination delivered a reliable, affordable, and CPU-compatible RAG system suitable for local or limited environments.

---

### Main drawbacks that I faced
1. As I don't have powerful enough laptop sometimes it takes longer to get everything setup and search.
2. Sometimes search doesn't find relevant articles if rephrased too much.

