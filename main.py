# main.py
from batch_rag.app import RagApplication
from batch_rag.batch_processor import BatchProcessor
from evals.retrieval_evaluation import evaluate

if __name__ == "__main__":
    
    # Processing data: builds multimodal index
    batch = BatchProcessor()
    batch.build_index()
    
    # Extracting(scraping) + building multimodal index 
    # batch.run_all()
    
    # Metrics evaluator 
    evaluate()
    
    # Rag streamlit application
    app = RagApplication()
    app.run()
