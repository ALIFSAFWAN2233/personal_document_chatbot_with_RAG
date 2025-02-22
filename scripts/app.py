from typing import List, Literal
from fastapi import FastAPI
from RAG_Implementation import encode_query, retrieve_documents, augmentNgenerate
from LLM_loader import LLM_loader
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate


#initialize the server
app = FastAPI()


# Put in the clients 
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


base_prompt = """
    
    You are a helpful AI assistant of a personal document chatbot application. Your task is to help the user by searching specific details and summarizing documents such as agreements and legal paper from the given context.
  
    Give your answer in your response and make sure it follows these criteria:
        1. Generate answer in a helpful and friendly manner. 
        2. Exclude the context or this prompt from your answer.
        3. Format your answers as you see fit by including sections, headers, subheaders, bullet points.
        4. Please include the source you used in your response such as the name of the document. 
        
    """
# Define a message model with role and content
class Message(BaseModel):
    role: Literal["user", "ai"]  
    content: str


#Define the request 'model'
class QueryRequest(BaseModel):
    user_query: str
    messages: List[Message] #All the previous messages from both the ai and the user will be passed from front end


#FastAPI uses type hints to define the parameters of an endpoint function


"""
    endpoint for receiving user's questions and returns answers
"""

@app.post("/query/")
async def process_query(request: QueryRequest):

    base_prompt = """
    
    You are a helpful AI assistant of a personal document chatbot application. Your task is to help the user by searching specific details and summarizing documents such as agreements and legal paper from the given context.
  
    Give your answer in your response and make sure it follows these criteria:
        1. Generate answer in a helpful and friendly manner. 
        2. Exclude the context or this prompt from your answer.
        3. Format your answers as you see fit by including sections, headers, subheaders, bullet points.
        4. Please include the source you used in your response such as the name of the document. 
        
    """

    #Initialize messages list
    messages = []
    

    #get the user query sent by the client-side
    user_query = request.user_query  

    query_embedding = encode_query(user_query)

    chunks = retrieve_documents(query_embedding)

    #Format the messages list just like how in the chat are
    messages.append(SystemMessage(content=base_prompt)) #append the base prompt

    #Append all the previous messages to the list
    for msg in request.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            messages.append(AIMessage(content=msg.content))
    
    #Append the latest message
    messages.append(HumanMessage(content=f"Context:\n{chunks}\n\nUser's Query:\n{user_query}"))

    #inferencing
    response = augmentNgenerate(model, tokenizer, messages)

    #print the response 
    print("RESPONSE GENERATED: \n\n", response)

    #return the response to the client side

    return{"response": response}




#endpoint for receiving documents which needed to preprocessed and added to the vector database



#endpoint to retrieve the full content of the specific documents by its doc_id


