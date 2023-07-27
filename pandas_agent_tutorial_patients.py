
import os
import constants
import pandas as pd
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

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
os.chdir('/Users/fabian/Desktop/langchain/')



agent = create_csv_agent(OpenAI(temperature=0),
                         ['data/patients_all.csv','data/patients.csv'],
                         model="gpt-3.5-turbo-0613",
                         verbose=True)

#agent.agent.llm_chain.prompt.template = '\nYou are working with {num_dfs} pandas dataframes in Python named df1, df2, etc. '

# streamlit framework
st.title('GPT prompter based on input data from patients files')
st.write('This Bot is trained on the patient files in the data folder plus connects to GPT 3.5')
query = st.text_input('Write your question here')



# %% [markdown]
# 
# You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
# You should use the tools below to answer the question posed of you:
# 
# python_repl_ast: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.
# 
# Use the following format:
# 
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [python_repl_ast]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question
# 
# 
# This is the result of `print(df.head())`:
# {df}
# 
# Begin!
# Question: {input}
# {agent_scratchpad}

# %%
st.write(agent.run(query))
