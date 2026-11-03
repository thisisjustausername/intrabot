import asyncio

from chatbot.tools import query_database

result = asyncio.run(query_database.ainvoke(
    {'question': 'Was ist das Modulhandbuch des Bachelorstudiengangs Data Science. Dies wird benötigt, um alle Module dessen zu finden.'}))
"""

result = asyncio.run(query_database.ainvoke(
    {'question': 'Suche das Modul mit dem Modulcode ges-0001 und gib alle verfügbaren Informationen zurück, insbesondere in welchen Studiengängen oder Modulhandbüchern es angeboten wurde.'}))
    """

print(result)
