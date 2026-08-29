'''
A script to create a vector store from crawled pages using Ollama embeddings and Chroma.
'''

import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)

with open('crawled_pages_all.json', 'r') as f:
    data = json.load(f)

pgs = [i[0] for i in data]
if len(pgs) != len(set(pgs)):
    print('Duplicate pages found')
    clean_data = []
    cl_urls = []
    for i in data:
        if i[0] not in cl_urls:
            clean_data.append(i)
            cl_urls.append(i[0])
    data = clean_data
print(len(data))
with open('crawled_pages_all.json', 'w') as f:
    json.dump(data, f)

print(f"Number of unique pages: {len(data)}")

embeddings = OllamaEmbeddings(model="qwen3-embedding")

docs = []
save_docs = []
for entry in data:
    for chunk in splitter.split_text(entry[1]):
        save_docs.append((entry[0], chunk))
        doc = Document(
            page_content=chunk,
            metadata={"source": entry[0]}
        )
        docs.append(doc)

with open('docs_all.json', 'w') as f:
    json.dump(save_docs, f)


print(f"Number of documents: {len(docs)}")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db_all"
)
