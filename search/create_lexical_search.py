'''
A script to create a lexicographical index to search the data.
'''

# TODO: optimize memory: currently saving full doc for each chunk: a lot of redundant data

import multiprocessing
import os
from multiprocessing import Pool

import bm25s
import Stemmer
from dotenv import load_dotenv
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

res = db.select(
    table='data.website',
    type_of_answer=ANSWER_TYPE.LIST_ANSWER,
    columns=['url', 'markdown']
)

db.shutdown()

if res.is_error:
    raise Exception(f"Error fetching data from database: {res.error}")

data = [tuple(i.values()) for i in res.data]


def process_document(entry) -> list[tuple[int, str, str]]:
    '''
    Process a single document entry by splitting it into chunks and creating Document objects.

    Args:
        entry (tuple): A tuple containing the document ID and the full document text.
    Returns:
        list[tuple[int, str, str]]: A tuple containing a list of chunk data and a list of Document objects.
    '''
    doc_id = entry[0] # the url of the page
    full_doc = entry[1] # the markdown content of the page


    save_list = [(doc_id, chunk, full_doc) for chunk in splitter.split_text(full_doc)]

    return save_list

def process_all_documents(data, num_workers=None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_document, data)

        save_docs = []

        for saves in results:
            save_docs.extend(saves)

    return save_docs

print(f'Workers: {multiprocessing.cpu_count()}')
save_docs = process_all_documents(data)

contents = [{'url': i[0], 'content': f'URL: {i[0]}\n\n{i[1]}'} for i in save_docs if not i[0].startswith('https://www.uni-augsburg.de/en/')] # filter out english results

stemmer = Stemmer.Stemmer('german')

corpus_tokens = bm25s.tokenize([i['content'] for i in contents], stopwords='de', stemmer=stemmer)

retriever = bm25s.BM25(corpus=contents)
retriever.index(corpus_tokens)

retriever.save('uni_all_retriever', corpus=contents)
