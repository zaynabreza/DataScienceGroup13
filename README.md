# QuestSecure

Welcome to the QuestSecure repository! 🚀 This assistant is designed to automate the filling of questionnaires efficiently and securely.

## Description

QuestSecure is a tool that simplifies the process of handling questionnaires by automatically filling them out based on a given input. This repository contains all the necessary files to set up and run QuestSecure.

## Installation

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

## Support

For support, contact us through the repository issues or pull requests. Contributions to enhance QuestSecure are always welcome! 🌟

Thank you for using QuestSecure!
