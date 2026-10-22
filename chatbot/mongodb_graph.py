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
    return {"messages": [resp]}


EMBED_RE = re.compile(r'(")?EMBED_START\s*(.*?)\s*EMBED_END(?(1)")', re.DOTALL)
def to_embedding_call(query: str) -> str:
    return EMBED_RE.sub(lambda m: json.dumps(embeddings.embed_query(m.group(2).strip())), query)


def generate_query(state: MessagesState):
    """Generate MongoDB aggregation pipeline"""

    projection_prompt = '''Schließe Felder, die du nicht brauchen wirst, direkt aus. Limitiere die Anzahl der zurückgegebenen Suchergebnisse IMMER streng. NUR wenn du nach einem einzigen Element mittels der _id suchst, darfst du alle Felder zurückgeben.
mhbs: Füge bei Modulhandbüchern in der Collection mhbs IMMER eine Projektion hinzu, die NUR folgende Felder oder eine Auswahl folgender Felder enthält: "_hash", "path", "name", "start_semester", "faculties", "mhb_group", "description", "module_groups". Bei "modules_groups" darfst du, falls nötig ein lookup auf modules machen, dann darfst du allerdings von modules NUR die Felder "_id", "module_code", "name" auswählen mit einer Projektion. Schließe IMMER "embedding_name", "embedding_faculties", "embedding_description" aus!
modules: Füge bei Modulen in der Collection modules IMMER eine Projektion hinzu, die IMMER mindestens folgende Felder oder noch weitere Felder ausschließt: "_hash", "inhalt", "ziele", "workloads", "klausurbeschreibung", "voraussetzungen", "bestehensvoraussetzungen", "dozent", "international", "prüfungen", "embedding_name", "embedding_prerequisites", "embedding_success_requirements", "embedding_content", "embedding_lecturer", "embedding_exam_outline", "embedding_goals", "embedding_faculty_chair", "embedding_workloads".
exams: Bei Klausuren müssen "_hash", "embedding_name", "embedding_type", "embedding_duration", "embedding_preparation", "embedding_description", "embedding_frequency" ausgeschlossen werden, falls du bestimmte Felder aber nicht brauchst, solltest du sie zudem ausschließen.

Verwende für Suchen von Text fuzzy search oder vector Search, wobei es am besten ist, beides zu verwenden und dann auszusortieren. Schreibe als queryVector NIEMALS Zahlen. Schreibe stattdessen den Suchtext zwischen EMBED_START und EMBED_END, als String:
{"$vectorSearch": {"index": "modules_vector_index", "path": "embedding_inhalt", "queryVector": "EMBED_START maschinelles Lernen EMBED_END", "numCandidates": 100, "limit": 8}}
Der Suchtext darf keine Anführungszeichen enthalten.

Wichtig ist: Halte die Anfrage einfach und antworte SCHNELL, erfrage immer maximal 3 Ergebnisse, nur in Ausnahmen mehr. Schließe alle Felder, die mit embedding_ beginnen explizit aus der Projektion aus.'''

    llm_with = llm.bind_tools([tool["mongodb_query"]], tool_choice="any")
    print("Generating query with prompt:\n", projection_prompt)
    resp = llm_with.invoke(
        [{"role": "system", "content": MONGODB_AGENT_SYSTEM_PROMPT + '\n\n' + projection_prompt}] + state["messages"]
    )
    query = resp.tool_calls[0]["args"]["query"]
    query = to_embedding_call(query)
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
