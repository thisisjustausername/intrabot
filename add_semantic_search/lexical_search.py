import bm25s
import Stemmer

reloaded_retriever = bm25s.BM25.load('uni_all_retriever', load_corpus=True)
stemmer = Stemmer.Stemmer('german')


def search_bm25(query: str, k: int = 15) -> list[tuple[str, str, float]]:
    '''
    Search through the University of Augsburg website using bm25

    Args:
        query (str): what to search for
        k (int): number of results to return
    Returns:
        list[str, str, float]: a list of search results with url, content, similarity
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


if __name__ == '__main__':
    query = 'Unfall'
    for index, i in enumerate(search_bm25(query)):
        print(f"Rank {index+1} (score: {i[2]:.2f}): {i[0]}")
