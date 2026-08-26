'''
A script to create a vector store from crawled pages using Ollama embeddings and Chroma.
'''

import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


with open('crawled_pages.json', 'r') as f:
    data = json.load(f)

pgs = [i[0] for i in data]
if len(pgs) != len(set(pgs)):
    print('Duplicate pages found')

embeddings = OllamaEmbeddings(model="qwen3-embedding")

docs = []

for entry in data:
    dta = 'URL: ' + entry[0] + '\n\n' + entry[1]

    doc = Document(
        page_content=dta,
    )
    docs.append(doc)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db"
)
