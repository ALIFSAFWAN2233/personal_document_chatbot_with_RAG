from fastapi import FastAPI
from RAG_Implementation import RAGmain
from LLM_loader import LLM_loader
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

#initialize the server
app = FastAPI()

origins =[
    "http://127.0.0.1:5500",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# load the model on the server
model_loader = LLM_loader()
model,tokenizer = model_loader.get_model()


#Define the request 'model'
class QueryRequest(BaseModel):
    user_query: str


#FastAPI uses type hints to define the parameters of an endpoint function


"""
    endpoint for receiving user's questions and returns answers
"""

@app.post("/query/")
async def process_query(request: QueryRequest):

    #get the user query sent by the client-side
    user_query = request.user_query  

    #run inference on the query
    response = RAGmain(user_query)

    #print the response 
    print("RESPONSE GENERATED: \n\n", response)

    #return the response to the client side

    return{"response": response}




#endpoint for receiving documents which needed to preprocessed and added to the vector database



#endpoint to retrieve the full content of the specific documents by its doc_id




