# main.py
from batch_rag.app import RagApplication
from batch_rag.batch_processor import BatchProcessor

if __name__ == "__main__":
    #batch = BatchProcessor()
    #batch.build_index()
    
    app = RagApplication()
    app.run()
