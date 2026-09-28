'''
A script to create a vector store from crawled pages using Ollama embeddings and Chroma.
'''

# TODO: optimize memory: currently saving full doc for each chunk: a lot of redundant data

import json
import multiprocessing
from multiprocessing import Pool

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

def process_document(entry) -> tuple[list, list[Document]]:
    '''
    Process a single document entry by splitting it into chunks and creating Document objects.

    Args:
        entry (tuple): A tuple containing the document ID and the full document text.
    Returns:
        tuple[list, list[Document]]: A tuple containing a list of chunk data and a list of Document objects.
    '''
    doc_id = entry[0]
    full_doc = entry[1]

    save_list = []
    chunk_list = []
    for chunk in splitter.split_text(full_doc):
        chunk_data = (doc_id, chunk, full_doc)
        save_list.append(chunk_data)

        doc = Document(
            page_content=chunk,
            metadata={"source": doc_id, "full_doc": full_doc}
        )
        chunk_list.append(doc)

    return save_list, chunk_list

def process_all_documents(data, num_workers=None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_document, data)

        save_docs = []
        docs = []
        for saves, chunks in results:
            save_docs.extend(saves)
            docs.extend(chunks)

    return save_docs, docs

'''
for entry in data:
    for chunk in splitter.split_text(entry[1]):
        save_docs.append((entry[0], chunk, entry[1]))
        doc = Document(
            page_content=chunk,
            metadata={"source": entry[0], "full_doc": entry[1]}
        )
        docs.append(doc)
'''

print(f'Workers: {multiprocessing.cpu_count()}')
save_docs, docs = process_all_documents(data)

print("Saving files...")
with open('docs_all.json', 'w') as f:
    json.dump(save_docs, f)


print(f"Number of documents: {len(docs)}")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db_all"
)
