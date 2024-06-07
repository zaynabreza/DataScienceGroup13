from langchain.chains import RetrievalAugmentedGeneration
from langchain.retrieval import LangChainElasticsearch
from langchain.prompts import StandardPrompt
from langchain.schema import PromptConfig
from langchain.schema import PromptConfig, LanguageModelConfig, CohereModelConfig
from langchain.retrieval import BaseRetriever
import faiss
import numpy as np
import cohere


# Temporary code to test the RAG model, will be replaced with Elasticsearch code
class ListRetriever(BaseRetriever):
    def __init__(self, embeddings):
        self.embeddings = np.array(embeddings)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])  # L2 distance for similarity
        self.index.add(self.embeddings)

    def retrieve(self, query_embedding, top_k=3):
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [(i, self.embeddings[i], 1.0 / (1.0 + distances[0][j])) for j, i in enumerate(indices[0])]
    

def load_embeddings():
    embeddings_images = np.loadtxt('Data/embeddings/images_questions.txt') 
    embeddings_pdf = np.loadtxt('Data/embeddings/pdf_questions.txt')  
    embeddings_excel = np.loadtxt('Data/embeddings/excel_questions.txt')
    embeddings_excelans= np.loadtxt('Data/embeddings/excel_answers.txt')

    # Concatenate all embeddings
    all_embeddings = np.concatenate([embeddings_images, embeddings_pdf, embeddings_excel, embeddings_excelans], axis=0)
    return all_embeddings


def setup_rag(embeddings):


    # Connect to Elasticsearch (assuming it's already running and contains your embeddings)
    # retrieval = LangChainElasticsearch(
    #     host="localhost",  # Change this to your Elasticsearch host
    #     index_name="your_index_name"  # Specify the name of your Elasticsearch index
    # )

    # Initialize the retriever with embeddings
    retriever = ListRetriever(embeddings)


    co = cohere.Client('yU3vQ4wcV8gbMCvsHngfAa6OfVmnaMQy1YOpVwqF')
    
    lm_config = CohereModelConfig(model="large", cohere_client=co)

    # Set up the prompt configuration
    prompt_config = PromptConfig(
        pre_retrieval_prompt=StandardPrompt("Given the question: '{}', "),
        post_retrieval_prompt=StandardPrompt(" using the following relevant information: '{}'")
    )
    
    # Create the RAG chain
    rag = RetrievalAugmentedGeneration(
        retrieval=retriever,
        language_model_config=lm_config,
        prompt_config=prompt_config
    )

    return rag


embeddings = np.random.rand(100, 768)  # Example embeddings

# Example usage
rag = setup_rag(embeddings)
question = "What is the capital of France?"
query_embedding = np.random.rand(768)  # Simulate a query embedding
response = rag.run(question, query_embedding=query_embedding)
print("Response:", response)

if __name__ == "__main__":
    rag = setup_rag(embeddings)
