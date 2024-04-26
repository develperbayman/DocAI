import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import docx

def get_document_text(docs):
    text = ""
    for doc in docs:
        if doc.name.endswith('.pdf'):
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif doc.name.endswith('.docx'):
            docx_document = docx.Document(doc)
            for paragraph in docx_document.paragraphs:
                text += paragraph.text
        elif doc.name.endswith('.txt'):
            text += doc.read().decode()
    return text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model='gpt-4')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question, conversation_key):
    response = st.session_state.conversations[conversation_key]['chain']({'question': user_question})
    st.session_state.conversations[conversation_key]['history'] = response['chat_history']

    for i, message in enumerate(response['chat_history']):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="ChatPDF: ", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    st.header("Chat with multiple Docs :books:")

    with st.sidebar:
        st.subheader("Your documents")
        #pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        docs = st.file_uploader("Upload your documents here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                #raw_text = get_pdf_text(pdf_docs)
                raw_text = get_document_text(docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                # create conversation chain
                conversation_key = f"Conversation {len(st.session_state.conversations) + 1}"
                st.session_state.conversations[conversation_key] = {
                    'chain': get_conversation_chain(vectorstore),
                    'history': []
                }

        conversation_key = st.selectbox("Select Conversation:", options=list(st.session_state.conversations.keys()))

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question, conversation_key)


if __name__ == '__main__':
    main()
