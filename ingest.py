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


from langchain.document_loaders import PyPDFLoader, DirectoryLoader
#from langchain.document_loaders import WebBaseLoader





##loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
#loader = WebBaseLoader("https://www.quadratics.com/MLOPSimplified.html")
loader = DirectoryLoader('./data/', glob='*.pdf', loader_cls=PyPDFLoader)

data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)


# ##### Step 3: Store the custom content into a Vector DB (Chroma)

# In[3]:

from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings

#vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embed_model)
vectorstore.save_local('vectorstore/db_faiss')
