
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


"""
model_loader = LLM_loader()
model,tokenizer = model_loader.get_model()

def RAGmain(user_query):


    #encode the query
    query_embedding = encode_query(user_query)

    #get relevant chunks
    chunks = retrieve_documents(query_embedding)

    #include the chunks and the user's query (Augmentation) and generate response (Generation)
    response = augmentation(user_query,chunks,model,tokenizer)

    return response







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

def augmentation(user_query, chunks, model, tokenizer):
    
    base_prompt = """
    
    You are a helpful AI assistant of a personal document chatbot application. Your task is to help the user by searching specific details and summarizing documents such as agreements and legal paper from the given context.
  
    Give your answer in your response and make sure it follows these criteria:
        1. Generate answer in a helpful and friendly manner. 
        2. Exclude the context or this prompt from your answer.
        3. Format your answers as you see fit by including sections, headers, subheaders, bullet points.
        4. Please include the source you used in your response such as the name of the document. 
        
    """

    max_new_tokens = 256

    
    """
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
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, include_prompt_in_result = False)
    print("the original output: ", outputs)
    #Decode and return only the new assistant response

    response = tokenizer.decode(outputs[0], skip_special_tokens = True)

    #messages.append({"role": "assistant", "content": response})
    print("THE RESPONSE:\n\n\n",response)

    """
    #Try out langchain's chat template (since it may not need to format or something)

    messages = [
        SystemMessage(content=base_prompt),
        HumanMessage(content=f"Context:\n{chunks}\n\nUser's Query:\n{user_query}")
    ]

    chat_prompt = ChatPromptTemplate.from_messages(messages)

    formatted_prompt = chat_prompt.format()
    #print("Formatted Prompt:\n", formatted_prompt)

    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, return_full_text=False)
    llm = HuggingFacePipeline(pipeline = hf_pipeline)


    response = llm.invoke(formatted_prompt)
    return(response)


if __name__ == '__main__':
    RAGmain()

