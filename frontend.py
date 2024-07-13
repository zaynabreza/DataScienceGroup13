import streamlit as st
from src.fill_questionnaire import fill_Questionnaire, generate_answer
from os.path import basename
from PIL import Image
import base64

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def get_pdf_base64_string(filepath):
    with open(filepath, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    return base64_pdf

def download_button(filepath):
    if filepath is None:
        return
    with open(filepath, 'rb') as file:
        filetype = filepath.split('.')[-1]
        filename = "filled_"+basename(filepath)
        if filetype == 'pdf':
            pdf_data = file.read()
            st.download_button(
                label="Download your answers as PDF",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf"
            )
        base64_pdf = get_pdf_base64_string(filepath)
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Hydac GPT 🤖", page_icon="🤖")
   
    css_kode_mono = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Kode+Mono&display=swap');

        body {
            font-family: 'Kode Mono', monospace;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Kode Mono', monospace;
        }

        p, div, span, li, a {
            font-family: 'Kode Mono', monospace;
        }
        .question {
            color: #b6d7a8; /* A soft green */
            background-color: #293241; /* A dark desaturated blue */
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 5px;
        }

        .answer {
            color: #cad2c5; /* A light gray */
            background-color: #2b3945; /* A moderate blue */
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        </style>
    """
    st.markdown(css_kode_mono, unsafe_allow_html=True)
    st.title('Quest Secure 🤖')
    st.write("""
    ### Welcome to Quest Secure! 👋
    Your intelligent chatbot for filling out security questionnaires.

    **Instructions:**
    1. Enter your text or upload your questionnaire in DOCX or PDF format.
    2. Enter your Elasticsearch API Key and Password in the sidebar.
    3. Click the Submit button to start processing your questionnaire.

    Once the questionnaire is uploaded, it will be processed, and the results will be generated shortly.
    """)
    st.sidebar.image(Image.open('logo.png'), use_column_width=True)
    elasticsearch_api_key = st.sidebar.text_input('Elasticsearch API Key', type='password')
    elasticsearch_password = st.sidebar.text_input('Elasticsearch Password', type='password')

    with st.form('my_form',clear_on_submit=True):
        question = st.text_input(label="Type your question here",)
        uploaded_file = st.file_uploader("Upload a document", type=["docx", "pdf"])
        submitted = st.form_submit_button('Submit')
    if submitted:
            if uploaded_file is not None:
                save_location = uploaded_file.name
                with open(save_location, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("File Upload successful. Processing may take a while.")
                answers_file_path, responses, questions = fill_Questionnaire(save_location)
                download_button(answers_file_path)
                # Display questions and responses
                st.markdown("### Questions and Responses")
                for question, response in zip(questions, responses):
                    st.markdown(f"<div class='question'>Question: {question}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='answer'>Answer: {response}</div>", unsafe_allow_html=True)
            elif question:
                st.success("Please wait while we process your question.")
                answer = generate_answer(question)
                st.markdown(f"<div class='question'>Question: {question}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='answer'>Answer: {answer}</div>", unsafe_allow_html=True)


main()