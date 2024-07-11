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
import re
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors


def initializeIndex(embeddingModel="all-MiniLM-L6-v2.gguf2.f16.gguf"):
    load_dotenv()  # This loads the .env file at the application start
    password = os.getenv('passwd')
    api_key = os.getenv('api_key')

    embedding = GPT4AllEmbeddings(model_name=embeddingModel) # add parameters here

    cloud_id  = '01a9e8bc7d7e4b91affdfcc8b88e70dd:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDVjZDFmMGQ0NDJkMzQ3ODA5ZmNiMjk4MTM5NmE4NGMxJGRmNTZhYTExYmE2YzRlZWZhOTE4NTBkMDJjZTY2MDIx'
    elastic_vector_search = ElasticsearchStore(
        es_cloud_id=cloud_id,
        index_name="embeddings_index2",
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
        retriever = elastic_vector_search.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        # retriever = elastic_vector_search.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
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
    prompt = hub.pull("zbr/rag-prompt")

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


def get_Questions(filePath, llm):

    # if file ends in .pdf:
    if filePath.endswith(".pdf"):
        reader = PdfReader(filePath)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        
        # Pattern to capture questions ending with a question mark and possibly followed by text in brackets
        pattern = r'•\s*(.*?\?)(\s*\([^)]*\))?'
        raw_questions = re.findall(pattern, text, re.DOTALL)

        # Clean the questions and combine parts
        questions = []
        for parts in raw_questions:
            question = ''.join(parts)  # Join the question parts and any following bracketed text
            question = re.sub(r'[_\r]+', '', question).strip()  # Clean up underscores and carriage returns
            question = re.sub(r'\s+', ' ', question)  # Replace multiple spaces with a single space
            questions.append(question)

        return questions



def fill_Questionnaire(file_path):



    questions = get_Questions(file_path, None)
    print("Questions extracted:")
    print(questions)


    # Initialize the index
    elastic_vector_search = initializeIndex()
    print("Index initialized")

     # Initialize the model
    # model_path = "/Users/I748655/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    model_path = "/Users/I748655/Library/Application Support/nomic.ai/GPT4All/mistral-7b-instruct-v0.1.Q4_0.gguf"
    llm, compressor = initializeModel(model_path)

    # Get the retriever
    retriever = getRetriever(elastic_vector_search, top_k=10)
    print("Retriever initialized")

    # Generate the response
    responses=[]

    for question in questions:
        response = generateResponse(retriever, model_path, question, llm)

        # response = ' '.join(list(dict.fromkeys(response.split())))

        responses.append(response)
    
    base_name = os.path.basename(file_path)  # Extract file name from path
    new_file_name = base_name.replace('.pdf', '_filled.pdf')

    c = canvas.Canvas(new_file_name, pagesize=letter)
    width, height = letter

    y_position = height - 40  # Start 40 pixels down from the top
    c.setFont("Helvetica-Bold", 14)

    for question, answer in zip(questions, responses):
        if y_position < 40:  # Check if we are near the bottom of the page
            c.showPage()
            y_position = height - 40  # Reset y_position for the new page
            c.setFont("Helvetica", 12)  # Reset font for the content
        
        # Print question
        c.setFont("Helvetica-Bold", 12)
        text = c.beginText(40, y_position)
        text.textLine(f"Question: {question}")
        c.drawText(text)

        # Adjust y position for answer
        y_position -= 20

        # Print answer
        c.setFont("Helvetica", 12)
        text = c.beginText(40, y_position)
        text.textLine(f"Answer: {answer}")
        c.drawText(text)

        # Adjust y position for next question
        y_position -= 40
    
    c.save()  # Save the PDF
    return new_file_name


def generate_answer(question):
    elastic_vector_search = initializeIndex()
    print("Index initialized")

     # Initialize the model
    # model_path = "/Users/I748655/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    model_path = "/Users/I748655/Library/Application Support/nomic.ai/GPT4All/mistral-7b-instruct-v0.1.Q4_0.gguf"
    llm, compressor = initializeModel(model_path)

    # Get the retriever
    retriever = getRetriever(elastic_vector_search, top_k=10)
    print("Retriever initialized")


    response = generateResponse(retriever, model_path, question, llm)

    return response

def main():
    # question = "Inwieweit wird in der Organisation Informationssicherheit gemanagt?"
    file_path = "/Users/I748655/Uni/Semester 2/Data Science/Project/DataScienceGroup13/questionnaires/SEC Questionaire 3.pdf"
    filled_file_path = fill_Questionnaire(file_path)
    print(f"Filled questionnaire saved at {filled_file_path}")
    
    


   




if __name__ == "__main__":
    main()