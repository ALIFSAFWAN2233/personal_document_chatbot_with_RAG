from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from LLM_loader import LLM_loader

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


"""


def main():

    model_loader = LLM_loader()
    model,tokenizer = model_loader.get_model()

    user_query = "Please summarize on what I should do to apply for CIMB credit card." # The user's query fetched from the front end

    #encode the query
    query_embedding = encode_query(user_query)

    #get relevant chunks
    chunks = retrieve_documents(query_embedding)

    #include the chunks and the user's query (Augmentation) and generate response (Generation)
    response = augmentation(user_query,chunks,model,tokenizer)


    print(response)







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

    print("This is the retrieved response from chromaDB: " , retrieved_response) # log for debugging

    for k, v in retrieved_response.items():
        if k == 'documents':
            for i in v:
                retrieved_chunks = i[0] + i[1] + i[2]
                print("This is the retrieved documents: ", retrieved_chunks) # log for debugging
    
    return retrieved_chunks



def augmentation(user_query, chunks, model, tokenizer):
    
    base_prompt = """
    
    You are a helpful AI assistant of a personal document chatbot application. Your task is to help the user by searching specific details and summarizing documents such as agreements and legal paper from the given context.
    
    Give your answer and make sure your answer follows these criteria:
        1. Generate answer in a friendly and helpful manner but concise. 
        2. Exclude the context or this prompt from your answer.
        3. Format your answers as you see fit by including headers, sections, and bullet points.
        4. Please include the source you used in your response such as the name of the document. 
        
    
    """

    max_new_tokens = 500

    #define the chat template
    messages = [
        {"role": "system", "content": base_prompt},       
         {"role": "user", "content": "Context:\n" + chunks + "\n\nUser's Query:\n" + user_query},
    ]


    # Format the conversation history for llama
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize 
    inputs = tokenizer(formatted_prompt,return_tensors="pt").to("cuda")
    
    # generate response
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    #Decode and return only the new assistant response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)


    return response

if __name__ == '__main__':
    main()

