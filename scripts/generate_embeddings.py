import os
import pandas as pd
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_elasticsearch import ElasticsearchStore
import numpy as np
from dotenv import load_dotenv
import os
from langchain.docstore.document import Document
import csv


def convert_excel_to_csv(directory):
    # Get a list of all Excel files in the directory
    excel_files = [file for file in os.listdir(directory) if file.endswith(('.xls', '.xlsx'))]
    for excel_file in excel_files:
        # Define the path to the Excel file
        excel_file_path = os.path.join(directory, excel_file)
        
        # Read all sheets from the Excel file
        excel_data = pd.read_excel(excel_file_path, sheet_name=None)

        # Process each sheet separately
        for sheet_name, df in excel_data.items():
            # Define the output CSV file path, including the sheet name
            output_csv_file = os.path.join(directory, f"{os.path.splitext(excel_file)[0]}_{sheet_name}.csv")

            # Drop unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # Drop rows where all values are NaN or columns that are entirely NaN
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)
            
            # Save the DataFrame to a CSV file
            df.to_csv(output_csv_file, index=False, encoding='utf-8')
            
            print(f"Converted {excel_file} - Sheet: {sheet_name} to {output_csv_file}")

def process_csv(file_path):
    docs = []
    with open(file_path, newline="", encoding='utf-8-sig') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        columns = csv_reader.fieldnames  # Get the column names dynamically
        for i, row in enumerate(csv_reader):
            to_metadata = {col: row[col] for col in columns if col in row}
            values_to_embed = {k: row[k] for k in columns if k in row}
            to_embed = "\n".join(f"{k.strip()}: {v.strip()}" for k, v in values_to_embed.items())
            newDoc = Document(page_content=to_embed, metadata=to_metadata)
            docs.append(newDoc)
    return docs

def load_documents(directory="././src/KnowledgeBase"):
    allDocs = {}

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print("Processing", filename)
            pdfLoader = PyPDFLoader(os.path.join(directory, filename))

            if "pdfs" not in allDocs:
                allDocs["pdfs"] = []

            allDocs["pdfs"].extend(pdfLoader.load())

        elif filename.endswith(".xlsx"):
            print("Processing", filename)
            excelLoader = UnstructuredExcelLoader(os.path.join(directory, filename))


            if "xlsx" not in allDocs:
                allDocs["xlsx"] = []

            allDocs["xlsx"].extend(excelLoader.load())
        
        # elif filename.endswith(".csv"):
        #     print("Processing", filename)
        #     csvloader = CSVLoader(os.path.join(directory, filename))

        #     if "csv" not in allDocs:
        #         allDocs["csv"] = []

        #     allDocs["csv"].extend(csvloader.load())
        else:
            continue

    print("Loaded", len(allDocs["pdfs"]), "pdf pages")
    print("Loaded", len(allDocs["xlsx"]), "excel sheets")

    
    return allDocs

def split_documents(allDocs):
    r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )

    r_splitter_excel = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=0,
        separators=["\r\n", "\n", "\t", ",", " "]
    )
    # splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=500, 
    #     chunk_overlap=0,
    #     length_function=len
    #     )

    splits={}

    for key in allDocs:
        splits[key] = r_splitter.split_documents(allDocs[key]) # splitting all the same way

    for key in allDocs:
        print("Number of splits for", key, ":", len(splits[key]))


    return splits

def generate_embeddings(splits, model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"):
    embedding = GPT4AllEmbeddings(model_name=model_name)

    load_dotenv()  # This loads the .env file at the application start
    password = os.getenv('passwd')
    api_key = os.getenv('api_key')

    cloud_id  = '01a9e8bc7d7e4b91affdfcc8b88e70dd:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDVjZDFmMGQ0NDJkMzQ3ODA5ZmNiMjk4MTM5NmE4NGMxJGRmNTZhYTExYmE2YzRlZWZhOTE4NTBkMDJjZTY2MDIx'
    elastic_vector_search = ElasticsearchStore(
        es_cloud_id=cloud_id,
        index_name="embeddings_index",
        embedding=embedding,
        es_user="group13",
        es_password=password,
        es_api_key=api_key
    )

    for key in splits:
        elastic_vector_search.add_documents(splits[key])

    print ("Added all documents to the index")

    return elastic_vector_search

def delete_index(elastic_vector_search, index_name="embeddings_index"):
    elastic_vector_search.delete(index_name)



def main():
    
    allDocs = load_documents()

    splits = split_documents(allDocs)

    elastic_vector_search = generate_embeddings(splits)

    
if __name__ == "__main__":
    main()