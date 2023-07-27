
import os
import sys
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import streamlit as st
import constants

# get api key
os.environ["OPENAI_API_KEY"] = constants.APIKEY

# streamlit framework
st.title('GPT prompter based on input data')
st.write('This Bot is trained on all .txt files in the data folder plus connects to GPT 3.5')
query = st.text_input('Write your question here')




loader = DirectoryLoader("data/", glob="*.txt")
data = loader.load()
index = VectorstoreIndexCreator().from_loaders([loader])
chain = ConversationalRetrievalChain.from_llm(
   llm=ChatOpenAI(model="gpt-3.5-turbo"),
   retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
 )


# loader2 = CSVLoader(file_path="data/patients.csv")
# data = loader2.load()
# index_creator = VectorstoreIndexCreator()
# docsearch = index_creator.from_loaders([loader2])
# chain2 = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
# response2 = chain2({"question": query2})

# chat_history = []
# if query2:
#   result = chain2({"question": query2, "chat_history": chat_history})
#   chat_history.append((query2, result['answer']))
#   st.write(result['answer'])




#this would print the input data on the site
#st.write(data)

chat_history = []
if query:
  result = chain({"question": query, "chat_history": chat_history})
  chat_history.append((query, result['answer']))
  st.write(result['answer'])

