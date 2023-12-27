#!/usr/bin/env python
# coding: utf-8

# # Retrieval Augmentation Generation (RAG) with LLAMA.CPP Quantized Model

# ### Install llama.cpp llama-cpp-python, chromadb
# In my previous video, I have shown how to build a quantized model from llama.cpp
# 
# In this notebook, you will see how to do RAG on a quantied model so that you can query your documents.
# 
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.64 --no-cache-dir
# 
# pip install chromadb 

# ##### Step 1: Instantiate an embed model which later will be used for storing data in the vector DB

# In[1]:

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = 'cpu'
embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)


# ##### Step 2: Process Custom Content into Chunks

# In[2]:


#from langchain.document_loaders import PyPDFLoader, DirectoryLoader
#from langchain.document_loaders import WebBaseLoader





##loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
#loader = WebBaseLoader("https://www.quadratics.com/MLOPSimplified.html")
#loader = DirectoryLoader('./data/', glob='*.pdf', loader_cls=PyPDFLoader)

#data = loader.load()

#from langchain.text_splitter import RecursiveCharacterTextSplitter

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#all_splits = text_splitter.split_documents(data)


# ##### Step 3: Store the custom content into a Vector DB (Chroma)

# In[3]:


#from langchain.vectorstores import Chroma
#from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import FAISS

#vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)
vectorstore = FAISS.load_local('vectorstore/db_faiss', embeddings=embed_model)

# ##### Step 4: Set bindings for LLAMA.CPP quantized model and instantiate the model

# In[4]:


from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
n_gpu_layers = 32  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# In[5]:


#llama = LlamaCppEmbeddings(model_path="/data/llama.cpp/models/llama-2-7b-chat/ggml-model-q4_0.bin")
llm = LlamaCpp(
    model_path="openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=10,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=False,
)


# ##### Step 5: Do a similarity search on the Vectordb to retrieve data related to the query

# In[7]:



# ##### Step 6: Create a RAG pipeline to contextualize with the custom data and Query

# In[10]:


from langchain.chains import RetrievalQA

rag_pipeline = RetrievalQA.from_chain_type( llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever())

def generate_response(query_txt):
    qa = RetrievalQA.from_chain_type( llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever())
    return qa.run(query_txt) 


# In[11]:


#rag_pipeline("what is the best treatment for covid 19")


# In[12]:


#rag_pipeline("what is Cart T ")


# In[13]:


#llm("what is current chemo treatment for liver cancer")


# In[ ]:


import streamlit as st

st.title('Patho.Ai Pathologist LLM Agent Version 1.9 ')

st.write('Hello User!')


result = []
print(st.session_state)
# Agent execution
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        try:
            response = generate_response(prompt)
            st.write(response)
        except:
            st.write("Please ask me again")
