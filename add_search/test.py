import json
import asyncio
'''
with open('crawled_pages_all.json', 'r') as f:
    data = json.load(f)

clean_data = []
urls = []

pgs = [i[0] for i in data]
print(len(pgs))
print(len(set(pgs)))
if len(pgs) != len(set(pgs)):
    print('Duplicate pages found')
for i in data:
    if i[0] in urls:
        continue
    clean_data.append(i)
    urls.append(i[0])
data = clean_data
del clean_data
print(len(data) == len(set(pgs)))

docs = []

data.sort(key=lambda x: len(x[1]), reverse=True)
print(data[0][1])
print(len(data[0][1]))
'''

from chatbot.tools import dirty_search

docs = asyncio.run(dirty_search.ainvoke({"query": "Informatik Bachelor", "k": 3}))
print(docs)
