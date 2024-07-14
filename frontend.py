import streamlit as st
from src.fill_questionnaire import fill_Questionnaire, generate_answer
from os.path import basename
from PIL import Image
import base64

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def download_button(filepath):
    if filepath is None:
        return
    with open(filepath, 'rb') as file:
        filetype = filepath.split('.')[-1]
        filename = "filled_" + basename(filepath)
        if filetype == 'pdf':
            pdf_data = file.read()
            st.download_button(
                label="Download your answers as PDF",
                data=pdf_data,
                file_name=filename,
                mime="application/pdf"
            )

def extract_filename(source_path):
        return source_path.split('/')[-1]


def main():
    st.set_page_config(page_title="Quest Secure 🤖", page_icon="🤖", layout="wide")

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
            color: #b6d7a8;
            background-color: #293241;
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 5px;
        }

        .answer {
            color: #cad2c5; /* A light gray */
            background-color: #1e3a39; /* A dark teal */
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 10px;
        }

        .sidebar .sidebar-content {
            background-color: #3c415c;
            color: white;
        }

        .sidebar .sidebar-content a {
            color: white;
        }

        .sidebar p, .sidebar div, .sidebar span, .sidebar li, .sidebar a {
            font-size: 14px; /* Smaller font size for sidebar */
        }

        .source {
        color: #888; /* A light gray for less prominence */
        font-size: 12px; /* Smaller font size for footnotes */
        padding-left: 10px; /* Indent slightly for aesthetic alignment */
    }

        </style>
    """
    st.markdown(css_kode_mono, unsafe_allow_html=True)

    st.title('Quest Secure 🤖')
    st.write("""
    ### Welcome to Quest Secure! 👋
    Your intelligent chatbot for filling out security questionnaires.
    """)

    st.sidebar.image(Image.open('logo2.png'), use_column_width=True)
    
    st.sidebar.title("Instructions")
    st.sidebar.write("""
    1. Enter your text or upload your questionnaire in DOCX or PDF format.
    2. Enter your Elasticsearch API Key and Password below.
    3. Click the Submit button to start processing your questionnaire.
    """)

    st.sidebar.title("Settings")

    elasticsearch_api_key = st.sidebar.text_input('Elasticsearch API Key', type='password')
    elasticsearch_password = st.sidebar.text_input('Elasticsearch Password', type='password')

    with st.form('my_form', clear_on_submit=True):
        question = st.text_input(label="Type your question here")
        uploaded_file = st.file_uploader("Upload a document", type=["docx", "pdf"])
        submitted = st.form_submit_button('Submit')
    
    if submitted:
        with st.spinner('Processing...'):
            if uploaded_file is not None:
                save_location = uploaded_file.name
                with open(save_location, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("File Upload successful. Processing may take a while.")
                answers_file_path, responses, questions, sources = fill_Questionnaire(save_location)
                download_button(answers_file_path)
                st.markdown("### Questions and Responses")
                for question, response, srcs in zip(questions, responses, sources):
                    st.markdown(f"<div class='question'>Question: {question}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='answer'>Answer: {response}</div>", unsafe_allow_html=True)
                    filenames = [extract_filename(src) for src in srcs]
                    source_text = ", ".join(filenames)  # Combine sources into a single string
                    if source_text:  # Only display if there are sources
                        st.markdown(f"<div class='source'>Sources: {source_text}</div>", unsafe_allow_html=True)
            elif question:
                st.success("Please wait while we process your question.")
                answer, sources = generate_answer(question)
                st.markdown(f"<div class='question'>Question: {question}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='answer'>Answer: {answer}</div>", unsafe_allow_html=True)
                filenames = [extract_filename(src) for src in sources]
                source_text = ", ".join(filenames)  # Combine sources into a single string
                if source_text:  # Only display if there are sources
                    st.markdown(f"<div class='source'>Sources: {source_text}</div>", unsafe_allow_html=True)


main()
