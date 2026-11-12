import os

import bm25s
import Stemmer  # type: ignore
from dotenv import load_dotenv
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

reloaded_retriever = bm25s.BM25.load('uni_all_retriever', load_corpus=True)
stemmer = Stemmer.Stemmer('german')

def search_bm25(query: str, k: int = 15) -> list[tuple[str, str, float]]:
    '''
    Search through the University of Augsburg website using bm25
    Returns url, a chunk of the webpage and a similarity score for each result

    Args:
        query (str): what to search for
        k (int): number of results to return
    Returns:
        list[str, str, float]: a list of search results with url, a chunk of the webpage, similarity
    '''
    query_tokens = bm25s.tokenize([query], stopwords='de', stemmer=stemmer)
    results, scores = reloaded_retriever.retrieve(query_tokens, k=3*k)
    # for result in results:
    #     print(result)

    non_dupl_urls = []
    non_dupl = []

    for i, sim in zip(results[0], scores[0]):
        if i['url'] in non_dupl_urls:
            continue
        non_dupl_urls.append(i['url'])
        non_dupl.append((i['url'], i['content'], sim))

    results = non_dupl[:k]
    return results
    # for doc, score in zip(results, scores):
    #     print(f'{score}: {doc['url']}')
        # print(f"Rank {i+1} (score: {score:.2f}): {data[doc]['url']}")


def get_full_page(url: str) -> str:
    '''
    Get the full webpage content of a url

    Args:
        url (str): the url to get the content from
    Returns:
        str: the full page content
    '''
    res = db.select(
        table='data.website',
        type_of_answer=ANSWER_TYPE.SINGLE_ANSWER,
        columns=['markdown'],
        conditions={'url': url}
    )
    if res.is_error:
        print(f"Error fetching data from database: {res.error}")
        return 'Error occurred querying the database!!!'
        # raise Exception(f"Error fetching data from database")
    data = res.data
    if data and len(data) > 0:
        return data['markdown']
    return 'No content found for this url, this url might be invalid'
    # return docs.get(url, 'No content found for this url, this url might be invalid')


if __name__ == '__main__':
    query = 'Unfall'
    for index, i in enumerate(search_bm25(query)):
        print(f"Rank {index+1} (score: {i[2]:.2f}): {i[0]}")
