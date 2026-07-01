import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from datetime import datetime

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Fixed chunk size and overlap (100 overlap on 100 size means 0 progress)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # Using the correct Gemini 3.5 Flash model
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key, pdf_names):
    # Load the already saved vector store instead of recreating it
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Save to history
    st.session_state.conversation_history.append((
        user_question, 
        response['output_text'], 
        "Google AI", 
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
        ", ".join(pdf_names)
    ))

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs (v1) :books:")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'processed_pdfs' not in st.session_state:
        st.session_state.processed_pdfs = []

    # Sidebar UI
    with st.sidebar:
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vansh-singla-384954271/) [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vanxhfr/)")
        
        st.title("Menu:")
        api_key = st.text_input("Enter your Google API Key:")
        st.markdown("Click [here](https://ai.google.dev/) to get an API key.")
        
        if not api_key:
            st.warning("Please enter your Google API Key to proceed.")
            
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs and api_key:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks) # Creates and saves the DB once
                    st.session_state.processed_pdfs = [pdf.name for pdf in pdf_docs]
                    st.success("Processing Complete! You can now ask questions.")
            else:
                st.warning("Please upload PDF files and enter API key.")

        if st.button("Reset Chat"):
            st.session_state.conversation_history = []

    # Main Chat UI
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question and api_key and st.session_state.processed_pdfs:
        user_input(user_question, api_key, st.session_state.processed_pdfs)
    elif user_question and not st.session_state.processed_pdfs:
        st.warning("Please upload and process PDF files first.")

    # Display Conversation History
    for question, answer, model_name, timestamp, pdf_name in reversed(st.session_state.conversation_history):
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)

    # Download Button
    if st.session_state.conversation_history:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()