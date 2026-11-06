import json
import os
import re
from typing import Literal

# TODO: remove this only for testing
import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_mongodb.agent_toolkit import MONGODB_AGENT_SYSTEM_PROMPT
from langchain_mongodb.agent_toolkit.database import MongoDBDatabase
from langchain_mongodb.agent_toolkit.toolkit import MongoDBDatabaseToolkit
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr
from pymongo import MongoClient


def log_request(request):
    print("REQUEST BODY:", request.content.decode())
# END REMOVAL

load_dotenv()

embeddings = OllamaEmbeddings(model='qwen3-embedding')
mdl = 'nvidia/Qwen3.8-27B-NVFP4'
mdl = 'nemotron-3.5-super'

llm = ChatOpenAI(
    http_client=httpx.Client(event_hooks={"request": [log_request]}),
    model=mdl,
    temperature=0.5,
    max_completion_tokens=4096,
    streaming=True,
    # reasoning=False,
    base_url='http://localhost:8000/v1',
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
    top_p=0.95,
    # presence_penalty=0,
    # min_p=0,
    api_key=SecretStr('not-a-real-key'),
)
base_uri = "mongodb://localhost:27017/"
username = "unia-search-ai"
password = os.getenv('MONGO_DB_UNIA_SEARCH_AI_PASSWORD')
auth_source = "unia"

# Construct the full URI
connection_uri = f"mongodb://{username}:{password}@localhost:27017/?authSource={auth_source}"


db = MongoDBDatabase.from_connection_string(
    connection_uri, database="unia"
)

client = MongoClient(connection_uri, appname='intrabot_text_to_mql_agent')

# Create toolkit and extract tools
toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
tool = {t.name: t for t in tools}

# Tool nodes for LangGraph
schema_node = ToolNode([tool["mongodb_schema"]], name="get_schema")
run_node = ToolNode([tool["mongodb_query"]], name="run_query")

def list_collections(state: MessagesState):
    """Deterministic node to list available collections"""
    call = {
        "name": "mongodb_list_collections",
        "args": {},
        "id": "abc",
        "type": "tool_call",
    }
    call_msg = AIMessage(content="", tool_calls=[call])
    resp = tool["mongodb_list_collections"].invoke(call)
    summary = AIMessage(f"Available collections: {resp.content}")
    return {"messages": [call_msg, resp, summary]}

def call_get_schema(state: MessagesState):
    """Prompt LLM to select and call schema tool"""
    llm_with = llm.bind_tools([tool["mongodb_schema"]], tool_choice="any")
    resp = llm_with.invoke(state["messages"])
    print('CALLING SCHEMA')
    print(resp.tool_calls[0])
    return {"messages": [resp]}


EMBED_RE = re.compile(r'(")?EMBED_START\s*(.*?)\s*EMBED_END(?(1)")', re.DOTALL)
def to_embedding_call(query: str) -> str:
    return EMBED_RE.sub(lambda m: json.dumps(embeddings.embed_query(m.group(2).strip())), query)


def generate_query(state: MessagesState):
    """Generate MongoDB aggregation pipeline"""

    projection_prompt = '''Schließe Felder, die du nicht brauchen wirst, direkt aus. Limitiere die Anzahl der zurückgegebenen Suchergebnisse IMMER streng. NUR wenn du nach einem einzigen Element mittels der _id suchst, darfst du alle Felder zurückgeben.

Verwende für Suchen von Text fuzzy search oder vector Search, wobei es am besten ist, beides zu verwenden und dann auszusortieren. Schreibe als queryVector NIEMALS Zahlen. Schreibe stattdessen den Suchtext zwischen EMBED_START und EMBED_END, als String:
{"$vectorSearch": {"index": "modules_vector_index", "path": "embedding_name", "queryVector": "EMBED_START maschinelles Lernen EMBED_END", "numCandidates": 100, "limit": 2}}
Der Suchtext darf keine Anführungszeichen enthalten.

Regeln für $project:
- Verwende in einem $project ENTWEDER nur Inklusion (Felder mit 1) ODER nur Exklusion (Felder mit 0). Mische beides NIEMALS (einzige Ausnahme: "_id": 0).

Regeln:
- Verwende immer $limit.
- Wenn du Felder mit viel Text wie Inhalt, Ziele, etc. zurückgibst, limitiere IMMER auf maximal 1 Ergebnis. Wenn du AUSSCHLIEßLICH Felder mit wenig Text zurückgibst, darfst du bis zu 3 Ergebnisse zurückgeben. Besonders bei Modulhandbüchern ist das wichtig. Auch bei Modulen.

Vorgehensweise:
    1. Überlege dir ein Konzept, wie du die Anfrage in MongoDB Aggregation Pipeline umsetzen kannst. Du darfst search, vectorSearch, lookups, und alle weiteren Möglichkeiten verwenden.
    2. Erstelle die Pipeline in MongoDB Aggregation Pipeline Syntax.
    3. Überprüfe, ob die Pipeline die Aufgabe erfüllt und ob sie korrekt ist. Wenn nicht, korrigiere sie.

WICHTIG: MISCHE NIEMALS 0 UND 1 im selben $project.
WICHTIG: VERWENDE NUR aggregate queries.
Wichtig ist: Halte die Anfrage einfach und antworte SCHNELL, erfrage immer maximal 3 Ergebnisse, nur in Ausnahmen mehr.'''

    llm_with = llm.bind_tools([tool["mongodb_query"]], tool_choice="any")
    print("Generating query with prompt:\n", projection_prompt)
    resp = llm_with.invoke(
        [{"role": "system", "content": MONGODB_AGENT_SYSTEM_PROMPT + '\n\n' + projection_prompt}] + state["messages"]
    )
    query = resp.tool_calls[0]["args"]["query"]

    # Replace any EMBED_START ... EMBED_END with the actual embedding vector
    print(f'FINAL QUERY BEFORE EMBEDDING: {query}')
    query = to_embedding_call(query)
    # Sanitize query by removing all embedding fields
    # TODO: remove embedding fields from output

    resp.tool_calls[0]["args"]["query"] = query
    return {"messages": [resp]}


def check_query(state: MessagesState):
    """Validate and sanitize generated query"""
    original = state["messages"][-1].tool_calls[0]["args"]["query"]
    resp = llm.bind_tools([tool["mongodb_query"]], tool_choice="any").invoke(
        [
            {"role": "system", "content": MONGODB_AGENT_SYSTEM_PROMPT + '\n\nReview this MongoDB query. If it is correct, call mongodb_query with EXACTLY the same query. Otherwise call it with a corrected version. Never write admin or schema commands like getCollectionNames.'},
            {"role": "user", "content": original},
        ]
    )
    resp.id = state["messages"][-1].id
    return {"messages": [resp]}


def need_checker(state: MessagesState) -> Literal[END, "check_query"]:
    """Conditional edge: run checker if tool call present"""
    return "check_query" if state["messages"][-1].tool_calls else END



def create_langgraph_agent_with_enhanced_memory():
    """Create custom LangGraph agent with LLM-powered summarizing checkpointer"""

    # Build the graph
    g = StateGraph(MessagesState)

    # Add nodes
    g.add_node("list_collections", list_collections)
    g.add_node("call_get_schema", call_get_schema)
    g.add_node("get_schema", schema_node)
    g.add_node("generate_query", generate_query)
    g.add_node("check_query", check_query)
    g.add_node("run_query", run_node)

    # Add edges - temp_create_final_answer goes directly to END
    g.add_edge(START, "list_collections")
    g.add_edge("list_collections", "call_get_schema")
    g.add_edge("call_get_schema", "get_schema")
    g.add_edge("get_schema", "generate_query")
    g.add_edge("generate_query", "run_query")
    # g.add_conditional_edges("generate_query", need_checker)
    # g.add_edge("check_query", "run_query")
    g.add_edge("run_query", END)
    # g.add_edge("temp_create_final_answer", END)  # Direct to END - checkpoints handle persistence

    # return g.compile(checkpointer=summarizing_checkpointer)
    return g.compile()
