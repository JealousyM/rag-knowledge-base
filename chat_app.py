# chat_app.py

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from data_processing import FAISS_INDEX_PATH

# Настройка стилей (остается без изменений)
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #00FFAA !important;
    }
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
Ты — эксперт-ассистент. Используй предоставленный контекст, чтобы ответить на вопрос. 
Если не уверен в ответе, скажи, что не знаешь. Будь кратким и фактологичным (максимум 3 предложения). Отвечай строго на русском языке.

Вопрос: {user_query} 
Контекст: {document_context} 
Ответ:
"""

EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")
LANGUAGE_MODEL = OllamaLLM(model="llama2")

# Загрузка индекса FAISS
vector_store = FAISS.load_local(FAISS_INDEX_PATH, EMBEDDING_MODEL, allow_dangerous_deserialization=True)

def find_related_documents(query):
    documents = vector_store.similarity_search(query)
    return documents

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    response = response_chain.invoke({"user_query": user_query, "document_context": context_text})
    return response

# UI Configuration
st.title("📘 RS.SCM AI")
st.markdown("### Твой АИ-ассистент по документам")
st.markdown("---")

st.success("✅ Документы успешно обработаны! Задавайте свои вопросы ниже.")

# Чат
user_input = st.chat_input("Введите ваш вопрос о документах...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.spinner("Анализирую документы..."):
        relevant_docs = find_related_documents(user_input)
        ai_response = generate_answer(user_input, relevant_docs)
    
    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)