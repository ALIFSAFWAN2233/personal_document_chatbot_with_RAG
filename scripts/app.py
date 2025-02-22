from fastapi import FastAPI
from RAG_Implementation import RAGmain
from LLM_loader import LLM_loader

#initialize the server
app = FastAPI()

# load the model on the server
#model_loader = LLM_loader()
#model,tokenizer = model_loader.get_model()

#get the user_query from the frontEnd

#FastAPI uses type hints to define the parameters of an endpoint function


"""
    endpoint for receiving user's questions and returns answers
"""

@app.get("/query/")
async def process_query(user_query: str): 

    #run inference on the query
    response = RAGmain(user_query)

    #print the response 
    print(response)

    #return the response to the client side

    return{"message": "Hello World"}




#endpoint for receiving documents which needed to preprocessed and added to the vector database



#endpoint to retrieve the full content of the specific documents by its doc_id




