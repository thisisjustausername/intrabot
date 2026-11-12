'''
A script to create a vector store from crawled pages using Ollama embeddings and Chroma.
'''

# TODO: optimize memory: currently saving full doc for each chunk: a lot of redundant data

import json
import multiprocessing
import os
from multiprocessing import Pool

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from postql import DB
from postql.types import ANSWER_TYPE, Credentials

load_dotenv()
creds = Credentials(
    user=os.environ.get("USERDB"), # type: ignore
    password=os.environ.get("POSTGRES_PASSWORD"), # type: ignore
    host=os.environ.get("POSTGRES_HOST"), # type: ignore
    port=int(os.environ.get("PORT")), # type: ignore
    database_name=os.environ.get("DBNAME") # type: ignore
)
db = DB(creds)

create_embeddings: bool = False

splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)

# Database already handles that
"""
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
"""

res = db.select(
    table='data.website',
    type_of_answer=ANSWER_TYPE.LIST_ANSWER,
    columns=['url', 'markdown']
)

if res.is_error:
    raise Exception(f"Error fetching data from database: {res.error}")

data = [tuple(i.values()) for i in res.data]


def process_document(entry) -> tuple[list, list[Document], tuple[str, str]]:
    '''
    Process a single document entry by splitting it into chunks and creating Document objects.

    Args:
        entry (tuple): A tuple containing the document ID and the full document text.
    Returns:
        tuple[list, list[Document], tuple[str, str]]: A tuple containing a list of chunk data and a list of Document objects.
    '''
    doc_id = entry[0] # the url of the page
    full_doc = entry[1] # the markdown content of the page

    compl_doc = (doc_id, full_doc)

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

    return save_list, chunk_list, compl_doc

def process_all_documents(data, num_workers=None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_document, data)

        save_docs = []
        docs = []
        full_docs = []

        for saves, chunks, compl_doc in results:
            save_docs.extend(saves)
            docs.extend(chunks)
            full_docs.append(compl_doc)

    return save_docs, docs, full_docs

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
save_docs, docs, full_docs = process_all_documents(data)

print('CREATING INDEX...')
import bm25s
import Stemmer
data = [{'url': i[0], 'content': f'URL: {i[0]}\n\n{i[1]}', 'full_doc': f'URL: {i[0]}\n\n{i[2]}'} for i in save_docs if not i[0].startswith('https://www.uni-augsburg.de/en/')] # add url to content anyways for better results and filter out english results

stemmer = Stemmer.Stemmer('german')

corpus_tokens = bm25s.tokenize([i['content'] for i in data], stopwords='de', stemmer=stemmer)


retriever = bm25s.BM25(corpus=data)
retriever.index(corpus_tokens)

retriever.save('uni_all_retriever', corpus=data)

print("Saving files...")

# Already fields (url, markdown) in database
# with open('full_docs.json', 'w') as f:
#     json.dump(full_docs, f)

# NOTE: Only temporarily needed for creating search index
with open('docs_all.json', 'w') as f:
    json.dump(save_docs, f)


print(f"Number of documents: {len(docs)}")
if create_embeddings:
    embeddings = OllamaEmbeddings(model="qwen3-embedding")
    print("Creating vector store...")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="chroma_db_all"
    )
