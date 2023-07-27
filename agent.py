
import os
import pandas as pd
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI



os.environ["OPENAI_API_KEY"] = constants.APIKEY
df = pd.read_csv('/data/patients_all.csv')

agent = create_csv_agent(OpenAI(temperature=0), 
                         '/data/patients_all.csv', 
                         verbose=True)