
from sentence_transformers import SentenceTransformer
import chromadb
from LLM_loader import LLM_loader
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
#from langchain.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline




"""
This script is a pipeline for RAG implementation for this chatbot app

STEP 1: Retrieval
* Get the user's query
* Retrieved relevant document from chromadb

STEP 2: Augment
* Place/format the prompt that have the user's query + retrieved document chunks

Step 3: Generation
* Generate the response based on the given context 
* Post-process the output given so that it can send the data to the front-end 



### Maintaining the chat history ###

Step 1: Initialize the messages list globally in the app.py
Step 2: Initialize the initial chats in the messages
Step 3: When user_query is passed from the frontend, use it to to generate encoding and get relevant chunks
Step 4: add the relevant information to the global messages list including the previous messages
Step 5: generate a response by passing the messages
Step 6: return the response to the client side

"""





def encode_query(query):
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    embedding = model.encode(query)

    return embedding



def retrieve_documents(query_embedding):
    vectorDB_path = "C:/Users/User/Documents/SideProject/personal_document_chatbot_with_RAG/data/vectorDB"
    client=chromadb.PersistentClient(path=vectorDB_path)
    collection=client.get_collection(name="document_collection")
    retrieved_response = collection.query(
        query_embeddings = query_embedding.tolist(), 
        n_results=3,
        include=["documents","metadatas"]
    )

    #print("This is the retrieved response from chromaDB: " , retrieved_response) # log for debugging

    for k, v in retrieved_response.items():
        if k == 'documents':
            for i in v:
                retrieved_chunks = i[0] + i[1] + i[2]
                #print("This is the retrieved documents: ", retrieved_chunks) # log for debugging
    
    return retrieved_chunks

'''
    Will do further experimentation and POCs on the best prompt for this project use case
'''

def augmentNgenerate( model, tokenizer, messages):
    max_new_tokens = 256

    chat_prompt = ChatPromptTemplate.from_messages(messages)

    formatted_prompt = chat_prompt.format()

    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, return_full_text=False)
    llm = HuggingFacePipeline(pipeline = hf_pipeline)


    response = llm.invoke(formatted_prompt)

    response = AIMessage(content=response)
  
    #returns only the response content
    return(response.content)




