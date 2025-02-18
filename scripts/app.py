from fastapi import FastAPI
from RAG_Implementation import encode_query, retrieve_documents, augmentation
from LLM_loader import LLM_loader

#initialize the server
app = FastAPI()

# load the model on the server
model_loader = LLM_loader()
model,tokenizer = model_loader.get_model()


"""
    endpoint for receiving user's questions and returns answers
"""

@app.post("/")
async def root():
    return{"message": "Hello World"}




#endpoint for receiving documents which needed to preprocessed and added to the vector database



#endpoint to retrieve the full content of the specific documents by its doc_id




