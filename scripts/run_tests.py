from langchain_community.document_loaders import PyPDFLoader
import os
import json
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_elasticsearch import ElasticsearchStore
import numpy as np
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import GPT4All
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def initializeIndex(embeddingModel="all-MiniLM-L6-v2.gguf2.f16.gguf"):
    load_dotenv()  # This loads the .env file at the application start
    password = os.getenv('passwd')
    api_key = os.getenv('api_key')

    embedding = GPT4AllEmbeddings(model_name=embeddingModel) # add parameters here

    cloud_id  = '01a9e8bc7d7e4b91affdfcc8b88e70dd:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDVjZDFmMGQ0NDJkMzQ3ODA5ZmNiMjk4MTM5NmE4NGMxJGRmNTZhYTExYmE2YzRlZWZhOTE4NTBkMDJjZTY2MDIx'
    elastic_vector_search = ElasticsearchStore(
        es_cloud_id=cloud_id,
        index_name="embeddings_index",
        embedding=embedding,
        es_user="group13",
        es_password=password,
        es_api_key=api_key
    )

    return elastic_vector_search

def getRetriever(elastic_vector_search, top_k=5, compressor=None):
    if compressor:
        compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=elastic_vector_search.as_retriever(search_type = "mmr", search_kwargs={"k": top_k})
        )
        return compression_retriever
    else:
        # retriever = elastic_vector_search.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        retriever = elastic_vector_search.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
        return retriever

def build_context(results):
    return "\n\n".join(result.page_content for result in results)

def initializeModel(model_path):
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model=model_path, callbacks=callbacks, verbose=True)
    compressor = LLMChainExtractor.from_llm(llm)
    return llm, compressor

def generateResponse(retriever, model_path, question, llm):
    # Says max 3 sentences, can change accoriding to the requirement
    prompt = hub.pull("rlm/rag-prompt")

    example_messages = prompt.invoke(
        {"context": "filler context", "question": "filler question"}
    ).to_messages()

    rag_chain = (
    {"context": retriever | build_context, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    response = ""

    for chunk in rag_chain.stream(question):
        # print(chunk, end="", flush=True)
        # also save in response
        response += chunk

    return response




def main():


    # Initialize the index
    elastic_vector_search = initializeIndex()
    print("Index initialized")

     # Initialize the model
    model_path = "/Users/I748655/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    llm, compressor = initializeModel(model_path)

    question = "Inwieweit wird in der Organisation Informationssicherheit gemanagt?"

    # Get the relevant documents
    retriever = getRetriever(elastic_vector_search)
    print("Retriever initialized")

    # Generate the response
    response = generateResponse(retriever, model_path, question, llm)

    print(response)

   




if __name__ == "__main__":
    main()