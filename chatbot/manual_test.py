import asyncio

from chatbot.tools import query_database

result = asyncio.run(query_database.ainvoke(
    {'question': 'Was ist das Modulhandbuch des Bachelorstudiengangs Data Science. Dies wird benötigt, um alle Module dessen zu finden.'}))

print(result)
