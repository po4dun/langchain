#https://github.com/yvann-hub/Robby-chatbot/blob/main/tuto_chatbot_csv.py

import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.vectorstores import FAISS, Chroma
import tempfile
import constants
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
import os
import sys

# get api key
os.environ["OPENAI_API_KEY"] = constants.APIKEY

loader = DirectoryLoader("data/", glob="*.txt")
data = loader.load()
index = VectorstoreIndexCreator().from_loaders([loader])
embeddings = OpenAIEmbeddings()
vectors = FAISS.from_documents(data, embeddings)
chain = ConversationalRetrievalChain.from_llm(
   llm=ChatOpenAI(model="gpt-3.5-turbo"),
   retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
 )

user_api_key = constants.APIKEY
os.environ["OPENAI_API_KEY"] = constants.APIKEY

# streamlit framework
st.title('GPT custom data chat')
st.write('This bot connects to GPT 3.5 and is trained on all .txt files in the data folder')

if True :

    def conversational_chat(query):

        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))

        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hi, how can I help you?"  ]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey !"]

    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()


    with container:
        with st.form(key='my_form', clear_on_submit=True):

            user_input = st.text_input("Write your questions:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversational_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")