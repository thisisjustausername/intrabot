
import os
from urllib.parse import urljoin, urlparse
from queue import Queue
from requests import Session
from bs4 import BeautifulSoup
import html_to_markdown as htm

from login.login import load_session

session = Session()
visited = set()
pages = []
load_session(session)  # Load the session with login cookies or headers
url = 'https://www.uni-augsburg.de/de/portal/intranet/arbeitsalltag/kommunikation/corporate-design/'

def normalize_url(base_url, href):
    '''Converts relative URLs to absolute and strips fragments (#anchor).'''
    if not href.startswith('https://'):
        href = urljoin(base_url, href)
    parsed = urlparse(href)
    clean_url = f'{parsed.scheme}://{parsed.netloc}{parsed.path}'.rstrip('/')
    if parsed.query:
        clean_url += f'?{parsed.query}'
    return clean_url

if url is None:
    exit(0)
print(url)
try:
    response = session.get(url, timeout=20, allow_redirects=True)
    content_type = response.headers.get('Content-Type', '')
    if response.status_code != 200 or 'text/html' not in content_type:
        print("ERROR")
    markdown = htm.convert(response.text, options=htm.ConversionOptions(exclude_selectors=['script', 'style', 'noscript', 'footer', 'nav']))
    pages.append((url, markdown.content))
    soup = BeautifulSoup(response.text, 'html.parser')
    page_url = response.url
    base_tag = soup.find('base', href=True)
    if base_tag:
        # If base_tag is itself relative, resolve it against page_url
        base_url = urljoin(page_url, base_tag['href'])
    else:
        base_url = page_url

    for a_tag in soup.find_all('a', href=True):
        link = normalize_url(base_url, a_tag['href'])
        ll = link.strip().lower()
        if link is None or not any(ll.startswith(i) for i in ['https://my.corebook.io/uni-augsburg']): continue
        if link not in visited:
            print(link)
            visited.add(link)
            url = link
            break
except Exception as e:
    print(f'Error crawling {url}: {e}')

print("BREAK TO NEXT ROUND")

try:
    response = session.get(url, timeout=20, allow_redirects=True)
    content_type = response.headers.get('Content-Type', '')
    if response.status_code != 200 or 'text/html' not in content_type:
        print(response.text)
        print("ERROR")
    markdown = htm.convert(response.text, options=htm.ConversionOptions(exclude_selectors=['script', 'style', 'noscript', 'footer', 'nav']))
    pages.append((url, markdown.content))
    soup = BeautifulSoup(response.text, 'html.parser')
    page_url = response.url
    base_tag = soup.find('base', href=True)
    if base_tag:
        # If base_tag is itself relative, resolve it against page_url
        base_url = urljoin(page_url, base_tag['href'])
    else:
        base_url = page_url
    print(base_url)
    for a_tag in soup.find_all('a', href=True):
        link = normalize_url(base_url, a_tag['href'])
        ll = link.strip().lower()
        print(f'Found link: {link}')
        if link is None or not any(ll.startswith(i) for i in ['https://www.uni-augsburg.de/de/portal/intranet/', 'https://brand-portal.uni-augsburg.de/', 'https://my.corebook.io/uni-augsburg']): continue
        if link not in visited:
            print(link)
            visited.add(link)
except Exception as e:
    print(f'Error crawling {url}: {e}')
