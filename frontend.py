import streamlit as st
from scripts.fill_questionnaire import fill_Questionnaire
from os.path import basename
from PIL import Image

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

def main():
    st.set_page_config(page_title="Hydac GPT 🤖", page_icon="🤖")
    # css = """
    #         <style>
    #         @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    #         body {
    #             font-family: 'Poppins', sans-serif;
    #         }

    #         h1, h2, h3, h4, h5, h6 {
    #             font-family: 'Poppins', sans-serif;
    #         }

    #         p, div, span, li, a {
    #             font-family: 'Poppins', sans-serif;
    #         }
    #         </style>
    #     """
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
        </style>
    """
    st.markdown(css_kode_mono, unsafe_allow_html=True)
    st.title('Hydac GPT 🤖')
    st.write("""
    ### Welcome to Hydac GPT! 👋
    Your intelligent Chatbot for security questionnaires.
    
    **Instructions:**
    1. Enter your text or upload your document in DOCX or PDF format.
    2. Enter your Elasticsearch API Key and Password in the sidebar.
    3. Click the Submit button to start processing your document.
    
    Once the document is uploaded, it will be processed, and the results will be generated shortly.
    """)
    st.sidebar.image(Image.open('logo.png'), use_column_width=True)
    elasticsearch_api_key = st.sidebar.text_input('Elasticsearch API Key', type='password')
    elasticsearch_password = st.sidebar.text_input('Elasticsearch Password', type='password')

    with st.form('my_form',clear_on_submit=True):
        question = st.text_input(label="Type your question here",)
        uploaded_file = st.file_uploader("Upload a document", type=["docx", "pdf"])
        submitted = st.form_submit_button('Submit')
    if submitted and uploaded_file is not None:
        save_location = uploaded_file.name
        with open(save_location, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File Upload successful. Processing may take a while")
        answers_file_path = fill_Questionnaire(save_location)
        download_button(answers_file_path)
    if submitted and question is not None:
        st.success("Please wait while we process your question.")
        answers_file_path = fill_Questionnaire(None,textInput=question)
        download_button(answers_file_path)


main()