# 🧠 Multimodal RAG System for The Batch

This is a multimodal Retrieval-Augmented Generation (RAG) system that enables intelligent search and answer generation over news articles from [The Batch](https://www.deeplearning.ai/the-batch/). The system integrates textual and visual data using FAISS for fast similarity search and OpenAI's GPT for high-quality answers.

---

## Features

- Search using **text + visual context** (multimodal retrieval)
- Generate context-aware answers using **GPT-3.5-turbo**
- Embeds and indexes both **text and images**
- Streamlit-based interactive UI
- Offline evaluation of retrieval performance
- Built with **Poetry**, **PyTorch**, **Hugging Face Transformers**, **FAISS**, and **OpenAI api**

---

## System Design & Reasoning

### Retrieval

- **Text Embeddings**: `all-MiniLM-L6-v2` (lightweight and fast)
- **Image Embeddings**: `openai/clip-vit-base-patch32`
- Combined into a single vector `[text || image]` and stored in a **FAISS** index

### Generation

- **OpenAI GPT-3.5-Turbo** is used for generating context-aware responses
- Queries are answered **only based on retrieved context** using a controlled prompt

### Query Pipeline

- User types a natural language query
- Query is encoded as `[text || text-as-image]` using CLIP
- FAISS retrieves top-3 closest matches from the multimodal index
- Answer is generated via OpenAI using article titles, summaries, and image captions

---

## Evaluation Results

Evaluation was done using 5 realistic queries and labeled relevant titles.

| Metric         | Result |
| -------------- | ------ |
| Top-1 Accuracy | 60%    |
| Top-3 Recall   | 60%    |

---

## Setup with Poetry

### 1. Clone the repo

```bash
git clone https://github.com/Zh3ka21/multimodal_rag_system.git
cd multimodal_rag_system
```

### 2. Install Poetry

```bash
pip install poetry
```

### 3. Install dependencies

```bash
poetry install
```

### 4. Activate the virtual environment

```bash
poetry shell
```

### Environment Variables

Create a `.env` file with your OpenAI key:

```env
OPENAI_API_KEY=sk-...
```

---

## Run the App

```bash
streamlit run main.py
```

## Optionals information

1. If you need to rebuild index you can uncomment `batch.build_index()`
2. If you need to run whole extracting and building index pipeline uncomment `batch.run_all()`. Strongly recommend not to do it, as pictures were picked manually as a result of inability to scrap them using scrapper.
3. If you are using VScode I recommend you press CTRL + Shift + P -> Developer: Reload Window. Don't forget to choose correct venv in VScode.

The app will launch the interactive UI at <http://localhost:8501>.

---

## Project Structure

```
backups/
batch_data/             # Stored FAISS index and metadata
batch_rag/
├── app.py              # Streamlit app logic
├── batch_processor.py  # Scraping, embedding, indexing
├── logger.py           # Logger utility
evals/
├── evaluate_retrieval.py  # Retrieval evaluation
├── eval_dataset.json      # Labeled test queries
logs/
tests/
├── test_batch_processor.py
.env
main.py                # Entrypoint to run the app
```

## Future improvements

1. Usage of better model
2. Sending more tokens to models for better understanding
