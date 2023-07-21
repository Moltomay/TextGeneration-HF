# Import os to set API key
import os
import requests

from dotenv import find_dotenv,load_dotenv
from transformers import pipeline
# Bring in streamlit for UI/app interface
import streamlit as st



load_dotenv(find_dotenv())

#Set HF API TOKEN
HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Create instance of HF GP2 LLM

pipe = pipeline("text-generation", model="gpt2")


st.title('🦜🔗 Happy Life Paper Agent')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')
	

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = pipe(prompt)
    #response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)
