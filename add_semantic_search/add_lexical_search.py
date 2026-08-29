import bm25s
import json
import Stemmer

with open('docs_all.json', 'r') as f:
    data = json.load(f)
data = [{'url': i[0], 'content': f'URL: {i[0]}\n\n{i[1]}'} for i in data if not i[0].startswith('https://www.uni-augsburg.de/en/')] # add url to content anyways for better results and filter out english results

stemmer = Stemmer.Stemmer('german')

corpus_tokens = bm25s.tokenize([i['content'] for i in data], stopwords='de', stemmer=stemmer)


retriever = bm25s.BM25(corpus=data)
retriever.index(corpus_tokens)

retriever.save('uni_all_retriever', corpus=data)

query = 'Unfall'
query_tokens = bm25s.tokenize(query, stemmer=stemmer)

results, scores = retriever.retrieve(query_tokens, k=10)

for i in range(results.shape[1]):
    doc, score = results[0, i], scores[0, i]
    print(f"Rank {i+1} (score: {score:.2f}): {data[doc]['url']}")
