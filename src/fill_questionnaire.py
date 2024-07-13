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
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
import textwrap
import copy
from langdetect import detect
from google.cloud import translate_v2 as translate
import pytesseract
from PIL import Image
from docx import Document
import textract



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


def get_Questions(filePath):

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

    elif filePath.endswith(".doc"):
        text = textract.process(filePath)
        text=text.decode('utf-8')


        raw_questions = []
        collect = False
        for line in text.split('\n'):
            if 'Antwort:' in line:
                collect = False
            if collect:
                if line.strip():  # Ensuring non-empty lines
                    raw_questions.append(line.strip())
            if 'Fragebogen' in line:  # This could be the marker to start collecting questions
                collect = True

        # Clean the questions and combine parts
        questions = []
        for question in raw_questions:
            question = re.sub(r'[_\r]+', '', question).strip()  # Clean up underscores and carriage returns
            question = re.sub(r'\s+', ' ', question)  # Replace multiple spaces with a single space
            questions.append(question)

        return questions

    elif filePath.endswith(".docx"):
        doc = Document(filePath)
        raw_questions = []
        for para in doc.paragraphs:
            text = para.text.strip()
            raw_questions.append(text)

        # Clean the questions and combine parts
        questions = []
        for question in raw_questions:
            question = re.sub(r'[_\r]+', '', question).strip()  # Clean up underscores and carriage returns
            question = re.sub(r'\s+', ' ', question)  # Replace multiple spaces with a single space
            questions.append(question)

        return questions




def generate_pdf(file_path, responses, questions):
    # pdfmetrics.registerFont(TTFont('Helvetica', 'Helvetica.ttf'))  # Ensure the font is available
    styles = getSampleStyleSheet()
    styleN = styles['Normal']
    styleN.fontName = 'Helvetica-Bold'
    styleN.fontSize = 12
    styleN.leading = 14

    base_name = os.path.basename(file_path)
    new_file_name = base_name.replace('.pdf', '_filled.pdf')
    
    c = canvas.Canvas(new_file_name, pagesize=letter)
    width, height = letter
    margin = inch * 0.75
    max_width = width - 2 * margin
    text_height = 12  # Approximate line height based on font size
    
    y_position = height - margin
    c.setFont("Helvetica", 12)
    
    for question, answer in zip(questions, responses):
        # Wrap and draw the question
        c.setFont("Helvetica-Bold", 12)
        wrapped_question = textwrap.fill(question, width=80)
        question_lines = wrapped_question.split('\n')
        for line in question_lines:
            if y_position < margin + text_height:
                c.showPage()
                y_position = height - margin
                c.setFont("Helvetica-Bold", 12) 
            c.drawString(margin, y_position, line)
            y_position -= text_height
        
        y_position -= text_height / 2  # Add a little space between question and answer
        
        # Wrap and draw the answer
        c.setFont("Helvetica", 12) 
        wrapped_answer = textwrap.fill(answer, width=80)
        answer_lines = wrapped_answer.split('\n')
        for line in answer_lines:
            if y_position < margin + text_height:
                c.showPage()
                y_position = height - margin
                c.setFont("Helvetica", 12)
            c.drawString(margin, y_position, line)
            y_position -= text_height
        
        y_position -= text_height  # Extra space before next Q&A
    
    c.save()
    print(f"PDF generated: {new_file_name}")
    return new_file_name

def fill_Questionnaire(file_path,textInput=None):


    if textInput is None:
        questions = get_Questions(file_path)
    else:
        questions = [textInput]
    print("Questions extracted:")
    print(questions)


    # Initialize the index
    elastic_vector_search = initializeIndex()
    print("Index initialized")

     # Initialize the model
    model_path = "/Users/I748655/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    # model_path = "/Users/I748655/Library/Application Support/nomic.ai/GPT4All/mistral-7b-instruct-v0.1.Q4_0.gguf"
    # model_path = "/Users/omeriqbal/Downloads/mistral-7b-instruct-v0.1.Q4_0.gguf"


    load_dotenv()
    model_path = os.getenv('model_path')

    llm, compressor = initializeModel(model_path)
    print("Model initialized")

    # Get the retriever
    retriever = getRetriever(elastic_vector_search, top_k=9)
    print("Retriever initialized")

    # Generate the response
    responses=[]

    for question in questions:
        response = generateResponse(retriever, model_path, question, llm)

        # response = ' '.join(list(dict.fromkeys(response.split())))

        responses.append(response)

    responses = post_process_responses(responses, questions)

    # save the responses as npy
    np.save("responses.npy", responses)
    np.save("questions.npy", questions)
    
    return generate_pdf(file_path, responses, questions), responses, questions


def generate_answer(question):


    elastic_vector_search = initializeIndex()
    print("Index initialized")

     # Initialize the model

    # defaut model path
    model_path = "/Users/I748655/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    # model_path = "/Users/I748655/Library/Application Support/nomic.ai/GPT4All/mistral-7b-instruct-v0.1.Q4_0.gguf"

    # read model_path from .env file
    load_dotenv()
    model_path = os.getenv('model_path')


    llm, compressor = initializeModel(model_path)
    print("Model initialized")

    # Get the retriever
    retriever = getRetriever(elastic_vector_search, top_k=10)
    print("Retriever initialized")


    response = generateResponse(retriever, model_path, question, llm)

    # response = "In der Organisation wird die Informationssicherheit durch das ISMS (Sicherheitsmanagement) gemanagt. Der Umzug von Informationen und Systemen wird durch eine interne Haustechnik-Abteilung durchgeführt, unterstützt von anderen Abteilungen wie der IT-Administration. Dokumenteneigenschaften werden in einem Protokoll dokumentiert und kontrolliert. Die Geschäftsführung hat die Gesamtverantwortung für die Informationssicherheit übernommen und erhält einen Management-Report jeden Monat, um den Umsetzungsstand der Maßnahme zu kontrollieren."

    responses=[]
    questions = []
    questions.append(question)
    responses.append(response)

    # return generate_pdf("question.pdf", responses, questions)
    return response

def post_process_responses(responses, questions):
    print("Post processing responses")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "././hybrid-unity-429117-d7-1182307a468e.json"

    load_dotenv()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('google_key_path')

    
    translate_client = translate.Client()
    
    trigger_words = {'Question', 'Answer', 'Context', 'Translation', '\nQuestion', '\nAnswer', 'Note', '\nContext', '\nTranslation', "Note", '\nNote', '(Note'}

    processed_responses = []
    for response, question in zip(responses, questions):
        # 1. Remove question repetition
        new_r = copy.deepcopy(response)
        if question in response:
            start_index = new_r.index(question) + len(question)
            new_r = new_r[start_index:].lstrip()  # Remove the question and leading whitespaces
        
        
        # 2. Remove any leftover sentences ending with a question mark
        new_r = re.sub(r'\s*\n.*\?\s*\n', '\n', new_r)

        # 3. Handle trigger words
        for trigger_word in trigger_words:
            if trigger_word in new_r:
                trigger_index = new_r.index(trigger_word)
                new_r = new_r[:trigger_index]
                break

        # 4. See if any sentence still not in same lanaguage as question, if only sentence, translate it
        try:
            question_lang = detect(question)
        except:
            question_lang = 'de'
        
        sentences = re.split(r'(?<=[.!?])\s+', new_r)
        filtered_sentences = []
        all_foreign = True

        for sentence in sentences:
            try:
                if detect(sentence) == question_lang:
                    filtered_sentences.append(sentence)
                    all_foreign = False
            except:
                continue 

        if filtered_sentences:
            new_r = ' '.join(filtered_sentences)
        else:
            #check if theres any alhpahbet  in the response at all
            if any(c.isalpha() for c in new_r):
                new_r = translate_client.translate(new_r, target_language=question_lang)["translatedText"]
        
        processed_responses.append(new_r)
        # print("Original Response: ", response)
        # print("Processed Response: ", new_r)
        # print("**********************")
    
    return processed_responses


def main():
    # question = "Inwieweit wird in der Organisation Informationssicherheit gemanagt?"
    # file_path = "/Users/I748655/Uni/Semester 2/Data Science/Project/DataScienceGroup13/questionnaires/SEC Questionaire 3.pdf"
    # file_path = "/Users/I748655/Uni/Semester 2/Data Science/Project/DataScienceGroup13/questionnaires/test.pdf"
    file_path = "questionnaires/SEC Questionaire 3.pdf"
   
    filled_file_path = fill_Questionnaire(file_path)
    # test_questions = ["Inwieweit sind Richtlinien zur Informationssicherheit vorhanden?",
    #               "Inwieweit wird in der Organisation Informationssicherheit gemanagt?",
    #               "Inwieweit sind die Verantwortlichkeiten für Informationssicherheit organisiert?",
    #               "Inwieweit werden Informationssicherheitsanforderungen in Projekten berücksichtigt?",
    #               "Inwieweit sind die Verantwortlichkeiten zwischen Organisations-fremden IT-Service-Anbietern und der eigenen Organisation definiert?",
    #               "Inwieweit werden Informationswerte (Assets) identifiziert und erfasst?",
    #               "Inwieweit werden Informationswerte hinsichtlich ihres Schutzbedarfs klassifiziert und gemanagt?",
    #               "Inwieweit wird sichergestellt, dass nur evaluierte und freigegebene organisationsfremde IT-Dienste zum Verarbeiten von Informationswerten der Organisation eingesetzt werden?",
    #               "Inwieweit werden Informationssicherheitsrisiken gemanagt?",
    #               "Inwieweit wird die Einhaltung der Informationssicherheit in Verfahren und Prozessen sichergestellt?",
    #               "Inwieweit wird das ISMS von einer unabhängigen Instanz überprüft?",
    #               "Inwieweit werden Informationssicherheitsereignisse verarbeitet?",
    #               "Inwieweit wird die Eignung von Mitarbeitern für sensible Tätigkeitsbereiche sichergestellt?",
    #               "Inwieweit werden alle Mitarbeiter zur Einhaltung der Informationssicherheit verpflichtet?",
    #               "Inwieweit werden Mitarbeiter über die Risiken beim Umgang mit Informationen geschult und sensibilisiert?",
    #               "Inwieweit ist mobiles Arbeiten geregelt?",
    #               "Inwieweit werden Sicherheitszonen für den Schutz von Informationswerten gemanagt?",
    #               "Inwieweit ist in Ausnahmesituationen die Informationssicherheit sichergestellt?",
    #               "Inwieweit ist der Umgang mit Informationsträgern gemanagt?",
    #               "Inwieweit ist der Umgang mit mobilen IT-Geräten und mobilen Datenträgern gemanagt?",
    #               "Inwieweit ist der Umgang mit Identifikationsmitteln gemanagt?",
    #               "Inwieweit wird der Zugang von Benutzern zu Netzwerkdiensten, IT-Systemen und IT-Anwendungen gesichert?",
    #               "Inwieweit werden Benutzerkonten und Anmeldeinformationen sicher verwaltet und angewandt?",
    #               "Inwieweit werden Zugriffsberechtigungen vergeben und gemanagt?",
    #               "Inwieweit wird die Nutzung kryptografischer Verfahren gemanagt?",
    #               "Inwieweit werden Informationen während der Übertragung geschützt?",
    #               "Inwieweit werden Änderungen gesteuert?",
    #               "Inwieweit sind die Entwicklungs- und Testumgebungen von den Produktivumgebungen getrennt?",
    #               "Inwieweit werden IT-Systeme vor Schadsoftware geschützt?",
    #               "Inwieweit werden Ereignisprotokolle aufgezeichnet und analysiert?",
    #               "Inwieweit werden Schwachstellen erkannt und behandelt?",
    #               "Inwieweit werden IT-Systeme technisch überprüft (Systemaudit)?",
    #               "Inwieweit wird das Netzwerk der Organisation gemanagt?",
    #               "Inwieweit wird Informationssicherheit bei neuen oder weiterentwickelten IT-Systemen berücksichtigt?",
    #               "Inwieweit sind Anforderungen an Netzwerkdienste definiert?",
    #               "Inwieweit ist die Rückgabe und das sichere Entfernen von Informationswerten aus Organisationsfremden IT-Diensten geregelt?",
    #               "Inwieweit sind Informationen in gemeinsam genutzten organisationsfremden IT-Diensten geschützt?",
    #               "Inwieweit wird die Informationssicherheit bei Auftragnehmern und Kooperationspartnern sichergestellt?",
    #               "Inwieweit ist Geheimhaltung beim Austausch von Informationen vertraglich vereinbart?",
    #               "Inwieweit wird die Einhaltung regulatorischer und vertraglicher Bestimmungen sichergestellt?",
    #               "Inwieweit wird der Schutz von personenbezogenen Daten bei der Umsetzung der Informationssicherheit berücksichtigt?"]

    # generated_responses = np.load("postprocessing/generated_responses_mmr_mistral.npy")
    
    # responses = post_process_responses(generated_responses, test_questions)

    # # save as npy
    # np.save("postprocessing/generated_responses_mmr_mistral_postprocessed.npy", responses)

    # # open and print to test
    # responses = np.load("postprocessing/generated_responses_mmr_mistral_postprocessed.npy")
    
    # for r in responses:
    #     print(r)
    #     print("**********************")


    # file_path = "/Users/I748655/Uni/Semester 2/Data Science/Project/DataScienceGroup13/questionnaires/Fragebogen - Buchführung und Logistiksystem.doc"
    # print(get_Questions(file_path))
    
    


   




if __name__ == "__main__":
    main()