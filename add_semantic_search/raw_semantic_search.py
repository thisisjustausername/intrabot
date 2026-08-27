'''
'''

import re

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rich import print as p
from rich.markdown import Markdown

################################################################
'''
Initialize vector database
'''
################################################################

embeddings = OllamaEmbeddings(model='qwen3-embedding')
db = Chroma(
            persist_directory='chroma_db',
            embedding_function=embeddings,
        )

################################################################
'''
Create tool
'''
################################################################


def search_intranet(query: str, k: int = 5, similarity_threshold: float = 0.1) -> list[tuple[str, float]]:
    '''
    Searches the internal intranet websites for relevant results.

    Args:
        query (str): Query containing the information to be searched.
        k (int): Number of relevant results to return. Default is 5.
        similarity_threshold (float): Minimum similarity score for a result to be considered relevant. Default is 0.1.
    Returns:
        str: List of relevant results containing URL and content.
    '''
    matches = db.similarity_search_with_relevance_scores(query, k=k, score_threshold=similarity_threshold)
    return [(match.page_content, score) for match, score in matches]


def search(query: str, k: int = 5, visualize: bool = True) -> list[str]:
    '''
    Searches the internal intranet websites for relevant results.

    Args:
        query (str): Query containing the information to be searched.
        k (int): Number of relevant results to return. Default is 5.
        visualize (bool): Whether to visualize the results using rich. Default is True.
    Returns:
        str: List of relevant results containing URL and content.
    '''
    results =  search_intranet(query=query, k=k)
    prefix = 'https://uni-augsburg.de'
    start = lambda x: f'DOCUMENT No. {x[0]+1} ({round(x[1] * 100, 2)}%)\n\n'
    results = [start((index, prob)) + re.sub(r'\[([^\n]+)\]\((\/[^\n]+)\)', rf'[\1]({prefix}\2)', res) for index, (res, prob) in enumerate(results)]
    result = '\n\n\n***\n***\n***\n\n\n'.join(results)

    if visualize:
        p(Markdown(result))

    return results

def search_urls(query: str, k: int = 5, similarity_threshold: float = 0.1) -> list[tuple[str, float]]:
    '''
    Searches the internal intranet websites for relevant results and returns only the URLs.

    Args:
        query (str): Query containing the information to be searched.
        k (int): Number of relevant results to return. Default is 5.
        similarity_threshold (float): Minimum similarity score for a result to be considered relevant. Default is 0.1.
    Returns:
        str: List of relevant results containing only the URLs and their probability.
    '''
    results = search_intranet(query=query, k=k, similarity_threshold=similarity_threshold)
    urls = [(res[5:].split('\n', 1)[0], prob) for (res, prob) in results]
    # urls = sorted(set(urls), key=urls.index)
    dedupl_urls = {}
    for url in urls:
        if url[0] not in dedupl_urls:
            dedupl_urls[url[0]] = url[1]
    return sorted([(k, v) for k, v in dedupl_urls.items()], key=lambda x: x[1], reverse=True)

def chat(output_amount: int = 5, k: int = 15, similarity_threshold: float = 0.1):
    '''
    Starts a chat session where the user can input queries and receive relevant URLs from the internal intranet websites.

    Args:
        output_amount (int): Number of relevant results to return. Default is 5.
        k (int): Number of relevant results to return. Default is 15.
        similarity_threshold (float): Minimum similarity score for a result to be considered relevant. Default is 0.1.
    '''
    if k < output_amount:
        raise ValueError(f'k must be greater than or equal to output_amount. k: {k}, output_amount: {output_amount}')
    k = 15
    similarity_threshold = 0.1
    output_amount = 5
    while True:
        query = input('Suche:       ')
        res = search_urls(query=query, k=k, similarity_threshold=similarity_threshold)[:output_amount]
        for index, url in enumerate(res):
                print(f'{index+1} ({round(url[1] * 100, 2)}%): {url[0]}')


if __name__ == "__main__":
    '''query = "Korruption"
    k = 5
    res = search_urls(query=query, k=k)
    for index, url in enumerate(res):
        print(f'{index+1}: {url}')'''
    chat()
