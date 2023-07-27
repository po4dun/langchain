
import os
import constants
import pandas as pd
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

# get api key
os.environ["OPENAI_API_KEY"] = constants.APIKEY
os.chdir('/Users/fabian/Desktop/langchain/')


df = pd.read_csv('data/patients_all.csv')


agent = create_csv_agent(OpenAI(temperature=0),
                        model="gpt-3.5-turbo-0613",
                        'data/patients_all.csv',
                        verbose=True)

agent.agent.llm_chain.prompt.template

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
agent.run("how many rows are there?")

# %%
agent.run("how many people are female?")

# %%
agent.run("what is the distribution of age in this sample? please use Median, mean, and interquartile range")

# %% [markdown]
# ## LangChain CSV Loader

# %%
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='data/patients_all.csv')
data = loader.load()

# %%
#custom chain etc


