# %% [markdown]
# # RAG using Langchain

# %% [markdown]
# ## Packages loading & import

# %%
!pip install langchain
!pip install langchain_community
!pip install langchain_huggingface
!pip install langchain_text_splitters
!pip install langchain_chroma
!pip install rank-bm25
!pip install huggingface_hub

# %%
import os
import json
import bs4
import nltk
import torch
import pickle
import numpy as np

from numpy.linalg import norm
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

# %%
nltk.download('punkt')
nltk.download('punkt_tab')

# %% [markdown]
# ## Hugging face login
# - Please apply the model first: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
# - If you haven't been granted access to this model, you can use other LLM model that doesn't have to apply.
# - You must save the hf token otherwise you need to regenrate the token everytime.
# - When using Ollama, no login is required to access and utilize the llama model.

# %%
from huggingface_hub import login

hf_token = ""
login(token=hf_token, add_to_git_credential=True)

# %%
!huggingface-cli whoami

# %% [markdown]
# ## TODO1: Set up the environment of Ollama

# %% [markdown]
# ### Introduction to Ollama
# - Ollama is a platform designed for running and managing large language models (LLMs) directly **on local devices**, providing a balance between performance, privacy, and control.
# - There are also other tools support users to manage LLM on local devices and accelerate it like *vllm*, *Llamafile*, *GPT4ALL*...etc.

# %% [markdown]
# ### Launch colabxterm

# %%
!pip install colab-xterm
%load_ext colabxterm

# %%
!curl -fsSL https://ollama.com/install.sh | sh

# %%
%xterm

# %%
"""
ollama serve
ollama pull llama3.2:1b
"""

# %% [markdown]
# ## Ollama testing
# You can test your Ollama status with the following cells.

# %%
# Setting up the model that this tutorial will use
MODEL = "llama3.2:1b" # https://ollama.com/library/llama3.2:3b
EMBED_MODEL = "jinaai/jina-embeddings-v2-base-en"

# %%
# Initialize an instance of the Ollama model
llm = Ollama(model=MODEL)
# Invoke the model to generate responses
response = llm.invoke("What is the capital of Taiwan?")
print(response)

# %% [markdown]
# ## Build a simple RAG system by using LangChain

# %% [markdown]
# ### TODO2: Load the cat-facts dataset and prepare the retrieval database

# %%
!wget https://huggingface.co/ngxson/demo_simple_rag_py/resolve/main/cat-facts.txt

# %%
with open('cat-facts.txt', 'r') as f:
    lines = f.readlines()

refs = [line.strip() for line in lines if line.strip()]

# %%
from langchain_core.documents import Document
docs = [Document(page_content=doc, metadata={"id": i}) for i, doc in enumerate(refs)]

# %%
# Create an embedding model
model_kwargs = {'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# %%
vector_store = Chroma.from_documents(
    docs, embedding=embeddings_model)
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})

# %% [markdown]
# ### Prompt setting

# %%
system_prompt = (
    "Answer the question accurately using the provided context."
    "Provide one to three sentences with concise and direct responses."
    "Ensure the answer fully addresses the query."
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# %% [markdown]
# - For the vectorspace, the common algorithm would be used like Faiss, Chroma...(https://python.langchain.com/docs/integrations/vectorstores/) to deal with the extreme huge database.

# %%
llm_model = Ollama(model=MODEL)
question_answer_chain = create_stuff_documents_chain(llm=llm_model, prompt=prompt)

chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)

# %%
# Question (queries) and answer pairs
# Please do not modify this cell.
queries = [
    "How much of a day do cats spend sleeping on average?",
    "What is the technical term for a cat's hairball?",
    "What do scientists believe caused cats to lose their sweet tooth?",
    "What is the top speed a cat can travel over short distances?",
    "What is the name of the organ in a cat's mouth that helps it smell?",
    "Which wildcat is considered the ancestor of all domestic cats?",
    "What is the group term for cats?",
    "How many different sounds can cats make?",
    "What is the name of the first cat in space?",
    "How many toes does a cat have on its back paws?"
]
answers = [
    "2/3",
    "Bezoar",
    "a mutation in a key taste receptor",
    ["31 mph", "49 km"],
    "Jacobson’s organ",
    "the African Wild Cat",
    "clowder",
    "100",
    ["Felicette", "Astrocat"],
    "four",
]

# %%
counts = 0
for i, query in enumerate(queries):
    response = chain.invoke({"input": query})
    print(f"Query: {query}\nResponse: {response['answer']}\n")
    # The following lines perform evaluations.
    # if the answer shows up in your response, the response is considered correct.
    if type(answers[i]) == list:
        for answer in answers[i]:
            if answer.lower() in response['answer'].lower():
                counts += 1
                break
    else:
        if answers[i].lower() in response['answer'].lower():
            counts += 1

# TODO5: Improve to let the LLM correctly answer the ten questions.
print(f"Correct numbers: {counts}")


