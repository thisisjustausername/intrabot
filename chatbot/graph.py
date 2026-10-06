'''


Run using: chainlit run chatbot/graph.py --host 127.0.0.1 --port 8001
Serve vllm as followed: python -m vllm.entrypoints.openai.api_server --model nvidia/Qwen3.8-27B-NVFP4 --max-model-len 131072 --port 8000 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --tensor-parallel-size 2
'''


import os
import warnings
from typing import Annotated, Literal
from urllib.parse import quote_plus
import asyncio

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core._api.beta_decorator import LangChainBetaWarning
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import SecretStr
from typing_extensions import TypedDict

from chatbot.tools import dirty_search, full_page, search_intranet
from mhbai.student_counselor.langgraph.tools import (
    search_studiengang,
    get_studiengang_modulhandbuch,
    get_modul,
    get_klausur,
    get_klausur_by_mongodb_id,
    get_modul_by_mongodb_id,
    get_modulhandbuch_by_mongodb_id,
    get_datum
)

# from rich.console import Console
# from rich.live import Live
# from rich.markdown import Markdown


warnings.filterwarnings('ignore', category=LangChainBetaWarning)

load_dotenv()
USERDB = os.getenv('USERDB')
HOST = os.getenv('HOST')
PORT = os.getenv('PORT')
DBNAME = os.getenv('DBNAME')
PASSWORD = quote_plus(os.getenv('PASSWORD', ''))

conninfo = f"postgresql+asyncpg://{USERDB}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"

mdl = 'qwen3.8:27b'
mdl = 'nvidia/Qwen3.8-27B-NVFP4'

################################################################
'''
Initialize vector database and model
'''
################################################################

embeddings = OllamaEmbeddings(model='qwen3-embedding')
'''
db = Chroma(
            persist_directory='chroma_db_all',
            embedding_function=embeddings,
        )
'''

# serve vllm as followed: python -m vllm.entrypoints.openai.api_server --model nvidia/Qwen3.8-27B-NVFP4 --max-model-len 131072 --port 8000 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --tensor-parallel-size 2
model = ChatOpenAI(
    model=mdl,
    temperature=0.5,
    max_completion_tokens=4096,
    streaming=True,
    # reasoning=False,
    base_url='http://localhost:8000/v1',
    extra_body={"top_k": 20},
    top_p=0.95,
    # presence_penalty=0,
    # min_p=0,
    api_key=SecretStr('not-a-real-key'),
)
'''
model = ChatOllama(
    model=mdl,
    temperature=0.5,
    num_predict=4096,
    # num_ctx=262144,
    num_ctx=131072,
    streaming=True
)
'''

################################################################
'''
Create workflow
'''
################################################################

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int


# Augment the LLM with tools
tools = [search_intranet, dirty_search, full_page, search_studiengang, get_studiengang_modulhandbuch, get_modul, get_klausur, get_klausur_by_mongodb_id, get_modul_by_mongodb_id, get_modulhandbuch_by_mongodb_id, get_datum] # , suche_uni_augsburg]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools, parallel_tool_calls=True)


system_prompt = '''Du bist ein hochpräziser Assistent für die Website Universität Augsburg.
Du berätst Mitarbeitende und hast Zugriff auf das Intranet und den öffentlichen Bereich der Universität Augsburg Website.
Du kannst NUR Fragen bezüglich des Intranets beantworten. Wenn Du eine Frage erhältst, verwende das Suchwerkzeug, um die Informationen zu finden.
Falls die Frage keine Suchergebnisse liefert und nichts mit dem Intranet zu tun hat, antworte mit 'Darüber habe ich leider keine Kenntnisse.'
Nutze das Suchwerkzeug bei Fragen zu Informationen aus dem Intranet.
Antworte auf Deutsch und in schönem Markdown-Format. Für Aufzählungen sind besonders Tabellen aber auch Listen erwünscht.
Du kannst mehrere Argumente pro Tool-Aufruf verwenden.

Regeln:
    - Verwende das Suchwerkzeug search_intranet, um Informationen zu finden
    - Wenn du zu einem Resultat von search_intranet die gesamte Seite benötigst, verwende das Tool full_page mit der URL zu dieser Seite.
    - Wenn search_intranet keine Ergebnisse liefert, verwende das Tool dirty_search, um Informationen
    - Antworte auf Deutsch und in schönem Markdown-Format
    - Du kannst mehrere Tools gleichzeitig aufrufen, um schneller an Antworten zu gelangen. Allerdings solltest du nur ein Tool pro Thema aufrufen, das heißt wenn du Thema A mit Thema B vergleichen sollst, kannst du gleichzeitig zu Thema A und Thema B suchen, allerdings pro Thema nur ein Tool aufrufen.
    - Entnehme dabei das Wissen aus der ANTWORT DES SEARCH-TOOLS
    - Gebe immer eine Antwort. Wenn du keine Informationen findest, teile dies in deiner Antwort mit.
    - Duze die Nutzer/in
    - Gebe nur Links der Universität Augsburg aus.
    - Verwende NIE Informationen, die nicht aus dem Search-Tool stammen. Wenn du keine Informationen findest, teile dies in deiner Antwort mit.
    - Teile die URLs, zu denen Du Informationen aus dem Search-Tool verwendest.
    - Gebe NIEMALS MongoDB-IDs aus.

Tools:
    - search_intranet: Durchsucht die internen Intranet-Websiten und die offiziellen Websiten der Universität Augsburg auf passende Ergebnisse. (Search-Tool)
    - full_page: Liefert die komplette und vollständige Seite zu einer URL, die von search_intranet zurückgegeben wurde. (Search-Tool)
    - dirty_search: Durchsucht die offiziellen Websiten der Universität Augsburg auf passende Ergebnisse. (Search-Tool)
        * Verwende dirty_search als Fallback, wenn search_intranet nichts findet.
        * dirty_search findet nur öffentlich zugängliche Websites der Universität Augsburg und ist geeignet für Queries, die Typos enthalten.
    - search_studiengang: Durchsucht die internen Informationskarten für Studiengängen nach Studiengangsinformationen, Inhalten, Zulassungsvoraussetzungen (NC) und weiteren studiengangsspezifischen Fragen
            Du kannst Informationen aus folgenden Bereichen zur Suche verwenden und diese sind immer in der Antwort enthalten:
                * Studiengangsname
                * Inhalt
                * Berufsperspektiven
                * Ziele
                * Regelstudienzeit
                * Teil- / Vollzeitstudium
                * Zulassungsmodus
                * Studienbeginn
                * Unterrichtssprache
                * gefordertes Deutschniveau
            NUR die aufgeführten Punkte sind in den Antworten enthalten.
            Es wird empfohlen, NUR DEN STUDIENGANGSNAMEN im Query zu suchen, je nach Anfrage können auch die anderen Bereiche abgefragt werden.
    - get_klausur: Gibt Informationen zu passenden Klausuren zurück.
            Du kannst Informationen aus folgenden Bereichen zur Suche verwenden und diese sind immer in der Antwort enthalten:
                * name: Name der Klausur
                * description: Beschreibung der Klausur
                * preparation: Vorbereitung auf die Klausur
                * type: Typ der Klausur z.B. mündlich, schriftlich, Hausarbeit, Seminararbeit, ...
                * duration: Dauer der Klausur
                * frequency: Häufigkeit der Klausur, z.B. einmal pro Semester, einmal pro Jahr, ...
                * deadline: Deadline der Klausur
                * graded: Ob die Klausur benotet ist
                * id: ID der Klausur, diese muss nicht eindeutig sein
                * portion_of_grade: Anteil an der Note der Klausur in dem verwendeten Modul
            Du musst immer mindestens einen semantischen Parameter (name, description, preparation, type, duration, frequency) angeben, um die Suche zu starten. Die anderen Parameter sind optional und können als Filter verwendet werden.
    - get_modul: Gibt Informationen zu passenden Modulen zurück.
        Du kannst Informationen aus folgenden Bereichen zur Suche verwenden und diese sind immer in der Antwort enthalten:
            * name: Name des Moduls
            * content: Inhalt des Moduls
            * goals: Ziele des Moduls
            * lecturer: Dozent des Moduls
            * prerequisites: Voraussetzungen des Moduls
            * faculty_chair: Lehrstuhl des Moduls
            * workloads: Arbeitsbelastung des Moduls
            * success_requirements: Erfolgsvoraussetzungen des Moduls
            * exam_outline: Prüfungsordnung des Moduls
            * mandatory: Ob das Modul verpflichtend ist
            * module_code: Modulcode des Moduls
            * ects: ECTS-Punkte des Moduls
            * available_semesters: Verfügbare Semester des Moduls
            * recommended_semester_span: Empfohlene Semesteranzahl des Moduls
            * languages: Sprachen des Moduls
            * international: Ob das Modul international ist
            * weekly_hours: Wöchentliche Stunden des Moduls
            * workload_hours: Arbeitsstunden des Moduls
            * exams: Prüfungen des Moduls
        Verwende immer mindestens einen semantischen Parameter (name, content, goals, lecturer, prerequisites, faculty_chair, workloads, success_requirements, exam_outline), um die Suche zu starten. Die anderen Parameter sind optional und können als Filter verwendet werden.
    - get_studiengang_modulhandbuch: Gibt das Modulhandbuch für einen bestimmten Studiengang zurück. Verwende diese Suche immer, wenn du Informationen zu Module oder dem Aufbau des Studiengangs benötigst.
        Um das aktuelle Modulhandbuch zu finden, kannst du start_semester als Filter verwenden. Die ersten 4 Ziffern sind das Anfangsjahr, die letzte Ziffer gibt an, ob es sich um das Wintersemester (1) handelt. So wird das Wintersemester 2026/2ß27 beispielsweise zu 20261. Es wird empfohlen, das aktuelle Modulhandbuch zu verwenden oder nachzufragen, in welchem Semester das Studium gestartet wurde.
        Du kannst Informationen aus folgenden Bereichen zur Suche verwenden und diese sind immer in der Antwort enthalten:
            * name: Name des Modulhandbuchs
            * description: Ab wann man den Studiengang studieren kann. NICHT Inhalt des Studiengangs.
            * faculties: Fakultäten des Studiengangs
            * path: Pfad des Modulhandbuchs
            * start_semester: Startsemester des Modulhandbuchs. SEHR WICHTIG, du kannst einen Zeitraum, z.B. (20241, 20250) also WS 2024/25 bis SS 2025, oder ein einzelnes Semester, z.B. 20241, angeben.
            * k: Anzahl der Suchresultate. Es wird maximal der Wert 3 empfohlen, da die Dokumente sehr lang sind.
        Verwende immer mindestens einen semantischen Parameter (name, description, faculties, path), um die Suche zu starten. Der Parameter start_semester ist optional und kann als Filter verwendet werden.
        Rufe diese Suche nie mehrmals in einem ähnlichen Thema auf, da die Ergebnisse sehr ähnlich sind!!!
    - get_datum: Gibt das aktuelle Datum zurück. Du kannst das aktuelle Datum verwenden, um die Suche nach Studiengängen zu filtern, die in der Vergangenheit liegen oder in der Zukunft stattfinden.
    Verwende die folgenden Tools, um nach Klausuren, Modulen oder Studiengängen zu suchen, wenn du deren MongoDB-ID kennst (die MongoDB-IDs findest du durch die Tools get_klausur, get_modul oder get_studiengang_modulhandbuch)
    - get_modulhandbuch_by_mongodb_id: Gibt ausführliche Informationen zu einem Modulhandbuch anhand der MongoDB-ID zurück.
    - get_modul_by_mongodb_id: Gibt Informationen zu einem Modul anhand der MongoDB-ID zurück.
    - get_klausur_by_mongodb_id: Gibt Informationen zu einer Klausur anhand der MongoDB-ID zurück.
'''
# - Führe die Tools nur NACHEINANDER aus, nicht gleichzeitig. Warte auf die Antwort des Tools, bevor du das nächste Tool aufrufst.

# model node: decides whether to call the tool node
async def llm_call(state: dict):
    '''LLM decides whether to call a tool or not'''

    return {
        'messages': [
            await model_with_tools.ainvoke(
                [
                    SystemMessage(
                        content=system_prompt
                    )
                ]
                + state['messages']
            )
        ],
        'llm_calls': state.get('llm_calls', 0) + 1
    }


MAX_TOOL_CHARS = 100000
sem = asyncio.Semaphore(10)

async def run_tool(tool: dict):
    async with sem:
        try:
            out = await tools_by_name[tool['name']].ainvoke(tool['args'])
            if isinstance(out, Exception): raise out
            return out
        except Exception as e:
            out = f"Error occurred while running tool: {e}"
            return ToolMessage(content=out, tool_call_id=tool['id'], name=tool['name'])


async def tool_node(state: dict):
    '''Performs the tool call'''

    tool_calls = state['messages'][-1].tool_calls
    results = await asyncio.gather(*(run_tool(tool_call) for tool_call in tool_calls))
    # print(f'TOOL CALLS: {results}')
    return {'messages': list(results)}

"""
async def compact_messages(state: MessagesState) -> MessagesState:
    '''
    Compact messages to avoid exceeding the maximum token limit and keeping the chatbot performant

    Args:
        state (MessagesState): The current state of the messages
    Returns:
        MessagesState: The compacted state of the messages
    '''

    messages = state['messages']
    last_user_message_index = max(index for index, msg in enumerate(messages) if isinstance(msg, HumanMessage))
    updates = []
    for m in messages[:last_user_message_index]:
        if isinstance(m, ToolMessage) and not m.additional_kwargs.get('compacted', False):
            updates.append(ToolMessage(
                id=m.id,
                tool_call_id=m.tool_call_id,
                name=m.name,
                content=
            ))
"""


async def should_continue(state: MessagesState) -> Literal['tool_node', END]:
    '''Decide if we should continue the loop or stop based upon whether the LLM made a tool call'''

    messages = state['messages']
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return 'tool_node'

    # Otherwise, we stop (reply to the user)
    return END


################################################################
'''
Build agent
'''
################################################################

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node('llm_call', llm_call)
agent_builder.add_node('tool_node', tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, 'llm_call')
agent_builder.add_conditional_edges(
    'llm_call',
    should_continue,
    ['tool_node', END]
)
agent_builder.add_edge('tool_node', 'llm_call')

checkpointer = MemorySaver()

# Compile the agent
agent = agent_builder.compile(checkpointer=checkpointer)

query = 'Wo finde ich Informationen zu Korruption?'

def get_info(event) -> dict[str, str | None]:
    info = {}
    start = event.get('params', {}).get('data', ({},))
    if isinstance(start, tuple) and len(start) > 0:
        a = start[0].get('content', {})
        b = start[0].get('delta', {})
        info['type'] =  (a or b or {}).get('type', None)
        info['content'] = (a or b or {}).get('text', None)
        info['event'] = start[0].get('event', None)
    else:
       info['type'] = None
       info['content'] = None
       info['event'] = None
    return info


'''
async def main():
    console = Console()
    accumulated_text = ''
    status = False
    run = await agent.astream_events(
            {'messages': [HumanMessage(content=query)]},
            version='v3'
        )
    print('🧠 Lass mich kurz nachdenken...')
    with Live(Markdown(''), console=console, refresh_per_second=15) as live:
        async for event in run:
            # Filter for the actual chat model streaming event
            res = get_info(event)
            if res['type'] == 'tool_call':
                print(f'🛠️ Tool-Aufruf {event['params']['data'][0]['content']['name']}: {a if not 'query' in (a := event['params']['data'][0]['content']['args']) else a['query']}')
            if res['type'] in ['text', 'text-delta'] and event['params']['data'][1]['langgraph_path'][1] == 'llm_call' and res['event'] != 'content-block-finish':
                if status is False:
                    print()
                status = True
                accumulated_text += res['content'] if res['content'] is not None else ''
                live.update(Markdown(accumulated_text))
'''


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=conninfo)

@cl.on_chat_start
async def start():
    cl.user_session.set("thread_id", cl.context.session.id)

@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    msg = cl.Message(content="")
    await msg.send()
    last_run_id = None

    run = await agent.astream_events(
        {"messages": [HumanMessage(content=message.content)]},
        version="v3",
        config=config
    )
    async for event in run:
        res = get_info(event)

        if res['type'] == 'tool_call':
            tool_name = event['params']['data'][0]['content']['name']
            args = event['params']['data'][0]['content']['args']
            arg_display = args.get('query', args)

            async with cl.Step(name=f"🛠️ {tool_name}", type="tool") as step:
                step.input = str(arg_display)
            last_run_id = None

        if res['type'] in ['text', 'text-delta'] \
                and event['params']['data'][1]['langgraph_path'][1] == 'llm_call' \
                and res['event'] != 'content-block-finish':
            current_run_id = event['params']['data'][1].get('run_id')

            if current_run_id != last_run_id and msg.content and not msg.content.endswith(('\n', ' ')):
                await msg.stream_token('\n\n')

            last_run_id = current_run_id
            if res['content']:
                await msg.stream_token(res['content'])
    await msg.update()

@cl.on_chat_end
async def end():
    thread_id = cl.user_session.get("thread_id")
    if thread_id and thread_id in checkpointer.storage:
        checkpointer.storage.pop(thread_id)
