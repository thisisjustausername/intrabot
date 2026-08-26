'''


Run using: chainlit run add_semantic_search/graph.py --host 127.0.0.1 --port 8001
'''


import operator
import re
import warnings
from typing import Annotated, Literal
from urllib.parse import urljoin

import chainlit as cl
import httpx
import trafilatura
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core._api.beta_decorator import LangChainBetaWarning
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from typing_extensions import TypedDict

warnings.filterwarnings('ignore', category=LangChainBetaWarning)


mdl = 'qwen3.8:27b'

################################################################
'''
Initialize vector database and model
'''
################################################################

embeddings = OllamaEmbeddings(model='qwen3-embedding')
db = Chroma(
            persist_directory='chroma_db',
            embedding_function=embeddings,
        )

model = ChatOllama(
    model=mdl,
    temperature=0.5,
    num_predict=4096,
    num_ctx=262144,
    streaming=True
)

################################################################
'''
Create tools
'''
################################################################


@tool
async def search_intranet(query: str, k: int = 5) -> list[str]:
    '''
    Durchsucht die internen Intranet-Websiten auf passende Ergebnisse.
    Du kannst immer nur nach EINEM Aufruf pro Anfrage suchen. Für mehrere Aufrufe stelle mehrere Anfragen. Stelle die Anfragen NACHEINANDER, sonst treten Fehler auf. Sende immer nur eine Suchanfrage und warte auf die Antwort bevor du die nächste Anfrage sendest.

    Args:
        query (str): Die Suchanfrage, die Informationen oder eine Frage enthält.
        k (int): Die Anzahl der zurückzugebenden relevanten Ergebnisse.

    Returns:
        str: Die relevantesten Informationen aus der Website inklusive der Quelle (URL) am Anfang der Dokumente.
    '''
    matches = db.similarity_search(query, k=k)

    if not matches:
        return ['Keine passenden Informationen gefunden.']
    return [match.page_content for match in matches]


def replacer(match):
    label, url = match.group(1), match.group(2)
    absolute = urljoin('https://www.uni-augsburg.de', url)
    return f'URL to {label}: {absolute}'


# TODO: instead of using trafilatula, convert to markdown
@tool
async def suche_uni_augsburg(query: str, k: int = 3) -> list[str]:
    '''
    Findet Seiten der Uni Augsburg mit Informationen zu dem Query.

    Args:
        query (str): Die Suchanfrage, die Informationen oder eine Frage enthält. Mache deutlich, dass sich das Query auf die Universität Augsburg bezieht.
        k (int): Die Anzahl der zurückzugebenden relevanten Ergebnisse. Empfohlen sind 5 bis 7, da die Rückgabe sonst sehr lang werden kann.
    Returns:
        list[str]: Die relevantesten Informationen von der Website der Universität Augsburg, die der Anfrage entsprechen. Wenn keine relevanten Informationen gefunden werden, wird eine entsprechende Nachricht zurückgegeben.
    '''
    res = []
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
            res = await client.get(
                'http://localhost:8888/search?q=',
                params={'q': f'site:uni-augsburg.de {query}', 'format': 'json'},
                headers={
                    "Accept": "application/json",
                }
            )
            res.raise_for_status()
    except httpx.HTTPError:
        return ["Fehler bei der Suche"]
    # TODO: load additional pages when results smaller than k
    res = [{k: v for k, v in i.items() if k in ['title', 'content', 'url']} for i in res.json().get('results', [])][:k]
    res = [r['url'] for r in res if re.match(r'^https://www\.([a-zA-Z0-9-]+\.)?uni-augsburg\.de(/.*)?$', r['url'])]
    if not res:
        return ['Keine passenden Informationen gefunden.']
    results = []
    for url in res:
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; MyAgent/1.0)"
                })
                response.raise_for_status()
        except httpx.HTTPError:
            continue

        text = trafilatura.extract(response.text, include_links=True)
        if not text:
            continue
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replacer, text)
        text = text.replace('@uni-auni-a.de', '@uni-a.de')
        # condensed_text = await _condense_text(text, query)
        results.append(f'Quelle: {url}\n{text}')
    if not results:
        return ['Keine passenden Informationen gefunden.']
    return results


################################################################
'''
Create workflow
'''
################################################################

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# Augment the LLM with tools
tools = [search_intranet] # , suche_uni_augsburg]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


system_prompt = '''Du bist ein hochpräziser Assistent für das Intranet der Universität Augsburg.
Du kannst NUR Fragen bezüglich des Intranets beantworten. Wenn Du eine Frage erhältst, verwende das Suchwerkzeug, um die Informationen zu finden.
Falls die Frage keine Suchergebnisse liefert und nichts mit dem Intranet zu tun hat, antworte mit 'Darüber habe ich leider keine Kenntnisse.'
Nutze das Suchwerkzeug bei Fragen zu Informationen aus dem Intranet.
Antworte auf Deutsch und in schönem Markdown-Format.

Regeln:
    - Verwende das Suchwerkzeug search_intranet, um Informationen zufinden
    - Antworte auf Deutsch und in schönem Markdown-Format
    - Führe die Tools nur NACHEINANDER aus, nicht gleichzeitig. Warte auf die Antwort des Tools, bevor du das nächste Tool aufrufst.
    - Entnehme dabei das Wissen aus der ANTWORT DES SEARCH-TOOLS
    - Gebe immer eine Antwort. Wenn du keine Informationen findest, teile dies in deiner Antwort mit.
    - Duze die Nutzer/in
    - Gebe nur Links der Universität Augsburg aus.
    - Verwende NIE Informationen, die nicht aus dem Search-Tool stammen. Wenn du keine Informationen findest, teile dies in deiner Antwort mit.
    - Teile die URLs, zu denen Du Informationen aus dem Search-Tool verwendest.

Tools:
    - search_intranet: Durchsucht die internen Intranet-Websiten auf passende Ergebnisse. (Search-Tool)
'''


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


async def tool_node(state: dict):
    '''Performs the tool call'''

    result = []
    for tool_call in state['messages'][-1].tool_calls:
        tool = tools_by_name[tool_call['name']]
        observation = await tool.ainvoke(tool_call['args'])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call['id']))
    return {'messages': result}


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

# Compile the agent
agent = agent_builder.compile()

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

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()
    last_run_id = None

    run = await agent.astream_events(
        {"messages": [HumanMessage(content=message.content)]},
        version="v3"
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
