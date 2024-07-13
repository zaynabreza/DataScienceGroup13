# QuestSecure

Welcome to the QuestSecure repository! 🚀 This assistant is designed to automate the filling of questionnaires efficiently and securely.

##  📖 Description

QuestSecure simplifies the handling of security questionnaires by:

- **Automated Filling**: Uses a Retrieval-Augmented Generation (RAG) system, built on a knowledge base of internal policies.
- **Enhanced Security**: Ensures the security of data by performing all processing and language model computations on the local hardware.
- **Ease of Setup**: Provides all necessary files and detailed instructions for a quick start.


## 🛠️ Installation 

Before running the application, you need to set up your environment. Here's how you can do it:

### Step 1: Clone the Repository 

Start by cloning this repository to your local machine:

```bash
git clone https://github.com/zaynabreza/DataScienceGroup13.git
cd DataScienceGroup13
```

### Step 2: Install Dependencies 

Install all required Python libraries using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```
### Step 3: GPU4All and Model Setup 

Install GPU4All and download the model you wish to use.

### Step 4: Environment Variables

Add the path of the downloaded model, Elastic Search API, path of the Google key, and password of the Elastic Search user to the `.env` file. Place this file in the `src` folder:

```plaintext
passwd="YOUR_PASSWORD_HERE"
api_key="YOUR_ELASTIC_SEARCH_API_KEY_HERE"
model_path="YOUR_MODEL_PATH_HERE"
google_key_path="YOUR_GOOGLE_KEY_PATH_HERE"
```

### Step 5: Run the Application

Execute the application using Streamlit:

```bash
streamlit run frontend.py
```

## 🧠 Building the Knowledge Base

QuestSecure can build a knowledge base from documents for its questionnaire-filling capabilities. Currently, the system supports PDF and Excel files.

### Step 1: Organize Your Documents

Place all the PDF and Excel files you want to include in the knowledge base into a single folder.

### Step 2: Update Environment Variables

Add the path to your documents folder to the `.env` file. Set the `knowledge_base_path` variable with your folder path:

```
knowledge_base_path="YOUR_PATH_HERE"
```
### Step 3: Generate Embeddings

Run the following command to generate embeddings from your documents, which will be used to power the knowledge retrieval features of QuestSecure:

```
python generate_embeddings.py
```
This process may take some time depending on the number and size of the documents.

## Support

For support, contact us through the repository issues or pull requests. Contributions to enhance QuestSecure are always welcome! 🌟

Thank you for using QuestSecure!
