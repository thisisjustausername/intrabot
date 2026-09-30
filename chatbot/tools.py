'''
Create tools for chatbot
'''

import re
from urllib.parse import urljoin

import html_to_markdown as htm
import httpx
from langchain.tools import tool

from search.lexical_search import get_full_page, search_bm25

options = htm.ConversionOptions(exclude_selectors=['script', 'style', 'noscript', 'footer', 'nav'])


def replacer(match):
    label, url = match.group(1), match.group(2)
    absolute = urljoin('https://www.uni-augsburg.de', url)
    return f'URL to {label}: {absolute}'


@tool
async def search_intranet(query: str, k: int = 5) ->str:
    '''
    Durchsucht die internen Intranet-Websiten auf passende Ergebnisse.
    Es werden Chunks der passenden Seiten zurückgegeben´. Falls Du zu einem Resultat, die gesamte Seite erhalten willst, rufe das Tool full_page mit dem URL zu dieser Seite auf.
    Du kannst immer nur nach EINEM Aufruf pro Anfrage suchen. Für mehrere Aufrufe stelle mehrere Anfragen. Stelle die Anfragen NACHEINANDER, sonst treten Fehler auf. Sende immer nur eine Suchanfrage und warte auf die Antwort bevor du die nächste Anfrage sendest.

    Args:
        query (str): Die Suchanfrage, die Informationen oder eine Frage enthält.
        k (int): Die Anzahl der zurückzugebenden relevanten Ergebnisse.

    Returns:
        str]: Die relevantesten Informationen aus der Website mit URL in der ersten Zeile, Ähnlichkeitswert in der zweiten Zeile und Inhalt darunter. Wenn keine relevanten Informationen gefunden wurden, wird eine entsprechende Nachricht angegeben. Suchresultate werden durch '\n\n---\n\n' getrennt.
    '''
    # matches = db.similarity_search(query, k=k)
    matches = [f'URL: {i[0]}\nSimilarity: {float(i[2])}\n\n{re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replacer, i[1])}' for i in search_bm25(query, k=k)]
    if not matches:
        return 'Keine passenden Informationen gefunden.'
    return '\n\n---\n\n'.join(matches)


@tool
async def full_page(url: str) -> str:
    '''
    Erhalte die komplette und vollständige Seíte zu einer URL, die von search_intranet zurückgegeben wurde. Verwende diese Funktion nur, wenn Du die gesamte Seite benötigst. Die URL muss von search_intranet stammen, sonst wird keine Seite gefunden.

    Args:
        url (str):Die URL für die Website

    Returns:
        str: Der komplette Seiteninhalt
    '''
    return get_full_page(url)


# TODO: instead of using trafilatula, convert to markdown
@tool
async def dirty_search(query: str, k: int = 3) -> str:
    '''
    Findet Seiten der Uni Augsburg mit Informationen zu dem Query.
    Verwende diese Suche nur als Fallback, wenn search_intranet nichts findet.
    Diese Suche greift nur auf den öffentlichen Teil der Website zu, funktioniert aber besonders gut, wenn Tippfehler in der Suchanfrage vorhanden sind.

    Args:
        query (str): Die Suchanfrage, die Informationen oder eine Frage enthält. Mache deutlich, dass sich das Query auf die Universität Augsburg bezieht.
        k (int): Die Anzahl der zurückzugebenden relevanten Ergebnisse. Empfohlen sind 5 bis 7, da die Rückgabe sonst sehr lang werden kann.
    Returns:
        str: Die relevantesten Informationen von der Website der Universität Augsburg, die der Anfrage entsprechen. Wenn keine relevanten Informationen gefunden werden, wird eine entsprechende Nachricht zurückgegeben. Suchresultate werden durch '\n\n---\n\n' getrennt.
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
        return "Fehler bei der Suche"
    # TODO: load additional pages when results smaller than k
    res = [{k: v for k, v in i.items() if k in ['title', 'content', 'url']} for i in res.json().get('results', [])][:k]
    res = [r['url'] for r in res if re.match(r'^https://www\.([a-zA-Z0-9-]+\.)?uni-augsburg\.de(/.*)?$', r['url'])]
    if not res:
        return 'Keine passenden Informationen gefunden.'
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

        if response.url != url and response.url:
            url = str(response.url)
        text = htm.convert(response.text, options=options).content
        print(text)
        if not text:
            continue
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', replacer, text)
        # text = text.replace('@uni-auni-a.de', '@uni-a.de')
        splitted = text.split('@')
        for i in range(1, len(splitted)):
            email_end, rest = splitted[i].split('.de', 1)
            # unia website always marks email domains thick so remove the non thick part (has to match the thick part)
            if len(email_end) > 3 and len(email_end) < 50 and (split := email_end.split('**', 3))[0] == split[1]:
                email_end = f'@{email_end.split('**', 3)[1]}.de'
            else:
                email_end = f'@{email_end}.de'
            splitted[i] = email_end + rest
        text = ''.join(splitted)
        # condensed_text = await _condense_text(text, query)
        results.append(f'Quelle: {url}\n{text}')
    if not results:
        return 'Keine passenden Informationen gefunden.'
    return '\n\n---\n\n'.join(results)
