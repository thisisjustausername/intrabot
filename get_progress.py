import chromadb

client = chromadb.PersistentClient(path="chroma_db_all")  # match your actual path
collection = client.get_collection("langchain")
print("Count:", collection.count())

