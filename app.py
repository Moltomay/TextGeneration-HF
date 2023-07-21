# Import os to set API key
import os
import requests

from dotenv import find_dotenv,load_dotenv
from transformers import pipeline
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

load_dotenv(find_dotenv())

#Set HF API TOKEN
HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Create instance of HF GP2 LLM

pipe = pipeline("text-generation", model="gpt2")
embedding = pipeline("feature-extraction", model="facebook/bart-large")

# Create and load PDF Loader
loader = PyPDFLoader('HappyLifePaper.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
#store = Chroma.from_documents(pages, embedding, collection_name='LifePaper')

# Create vectorstore info object - metadata repo?
#vectorstore_info = VectorStoreInfo(
#    name="LifePaper",
#    description="Paper about Happy Life",
#    vectorstore=store
#)
# Convert the document store into a langchain toolkit
#toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
#agent_executor = create_vectorstore_agent(
#    llm=pipe,
#    toolkit=toolkit,
#    verbose=True
#)
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
