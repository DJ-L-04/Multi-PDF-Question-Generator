import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
import re
import random

# Load the question generator model
question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")
questions=None
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text: 
            text += page_text
    return text

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_length=512):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_questions(text_chunks, num_questions):
    questions = []
    prompt_template = "You are a professor conducting an examination for your students. Generate possible questions from the following text. Text: "

    for chunk in text_chunks:
        prompt = prompt_template + chunk
        
        generated_responses = question_generator(
            prompt, max_new_tokens=100, num_return_sequences=min(10, num_questions * 2), num_beams=10
        )
        questions.extend([res['generated_text'] for res in generated_responses])

    return random.sample(questions, min(num_questions, len(questions)))

def process_pdfs(uploaded_files, num_questions):
    pdf_texts = []
    
    with ThreadPoolExecutor() as executor:
        pdf_texts = list(executor.map(lambda pdf: clean_text(extract_text_from_pdf(pdf)), uploaded_files))
    
    all_questions = []
    
    for text in pdf_texts:
        text_chunks = chunk_text(text)
        questions = generate_questions(text_chunks, num_questions)
        all_questions.append(questions)
    
    return all_questions

# Streamlit App Interface
st.set_page_config(page_title="Multi-PDF Question Generator", layout="wide")

# Sidebar for inputs
st.sidebar.header("Your documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
num_questions = st.sidebar.number_input("Number of questions per document", min_value=1, max_value=10, value=3)

st.title("Multi-PDF Question Generator :books:")
st.markdown("Upload your PDF documents, and this app will generate questions based on their content.")

# Add a placeholder for the questions
question_placeholder = st.empty()

if st.button("Generate Questions"):
    with st.spinner("Generating..."):
        if uploaded_files:
            questions = process_pdfs(uploaded_files, num_questions)
            question_placeholder.empty()  # Clear the placeholder

            for i, pdf_questions in enumerate(questions):
                st.subheader(f"Questions from PDF {i+1}:")
                for idx, q in enumerate(pdf_questions, start=1):
                    st.write(f"**Q{idx}:** {q}")
        else:
            st.warning("Please upload at least one PDF file.")

# Footer
st.markdown("---")
st.markdown("Click here to download the questions")

# Optional: Add a feature to download questions as a text file
if st.button("Download Questions"):
    if questions:
        questions_text = "\n".join([f"PDF {i+1}:\n" + "\n".join(pdf_questions) for i, pdf_questions in enumerate(questions)])
        st.download_button(label="Download Questions", data=questions_text, file_name="questions.txt", mime="text/plain")
    else:
        st.warning("Generate questions before downloading.")

# Add an informative footer
st.markdown("---")
st.markdown("Made by Devi Jeyasri")
