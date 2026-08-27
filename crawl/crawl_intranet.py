'''
This script implements a threaded web crawler that starts from a given URL and crawls all reachable pages within the same domain. It uses the `requests` library to make HTTP requests, `BeautifulSoup` to parse HTML, and `html_to_markdown` to convert HTML content to Markdown format. The crawler is designed to handle multiple threads for concurrent crawling, improving efficiency.
'''
import json

import os
from urllib.parse import urljoin, urlparse
import threading
from queue import Queue
from requests import Session
from bs4 import BeautifulSoup
import html_to_markdown as htm

from login.login import load_session

def crawl(session: Session, url: str) -> str:
    '''
    Crawl the given URL and return the response text.

    Args:
        session (Session): The requests session to use for making the request already logged in.
        url (str): The URL to crawl.

    Returns:
        str: The response text from the URL.
    '''
    response = session.get(url)
    response.raise_for_status()
    return response.text

class ThreadedCrawler:
    def __init__(self, session: Session, start_url: str, max_threads: int = 5, whitelisted_domains: list[str] | None = None, timeout: int = 30):
        self.start_url = start_url
        self.domain = urlparse(start_url).netloc

        self.queue = Queue()
        self.visited = set()
        self.lock = threading.Lock()
        self.session = session
        self.max_threads = max_threads
        self.options = htm.ConversionOptions(exclude_selectors=['script', 'style', 'noscript', 'footer', 'nav'])
        self.whitelisted_domains = whitelisted_domains if whitelisted_domains is not None else ['https://www.uni-augsburg.de/de/portal/intranet/', 'https://brand-portal.uni-augsburg.de/', 'https://my.corebook.io/uni-augsburg']
        self.timeout = timeout
        self.pages: list[tuple[str, str]] = []

    def normalize_url(self, base_url, href):
        '''Converts relative URLs to absolute and strips fragments (#anchor).'''
        if not href.startswith('https://'):
            href = urljoin(base_url, href)
        parsed = urlparse(href)
        clean_url = f'{parsed.scheme}://{parsed.netloc}{parsed.path}'.rstrip('/')
        if parsed.query:
            clean_url += f'?{parsed.query}'
        return clean_url


    def worker(self):
        while True:
            url = None
            try:
                url = self.queue.get()
                if url is None:
                    return
                print(url)
                response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                content_type = response.headers.get('Content-Type', '')
                if response.status_code != 200 or 'text/html' not in content_type:
                    continue
                markdown = htm.convert(response.text, options=self.options)
                self.pages.append((url, markdown.content))

                soup = BeautifulSoup(response.text, 'html.parser')
                page_url = response.url
                base_tag = soup.find('base', href=True)
                if base_tag:
                    # If base_tag is itself relative, resolve it against page_url
                    base_url = urljoin(page_url, base_tag['href'])
                else:
                    base_url = page_url

                for a_tag in soup.find_all('a', href=True):
                    link = self.normalize_url(base_url, a_tag['href'])
                    if link is None:
                        continue
                    ll = link.strip().lower()
                    if not any(ll.startswith(i) for i in self.whitelisted_domains): continue
                    with self.lock:
                        if link not in self.visited:
                            self.visited.add(link)
                            self.queue.put(link)
            except Exception as e:
                print(f'Error crawling {url}: {e}')
            finally:
                self.queue.task_done()

    def run(self):
        # Seed initial URL
        self.visited.add(self.start_url)
        self.queue.put(self.start_url)

        # Start worker threads
        threads = []
        for _ in range(self.max_threads):
            t = threading.Thread(target=self.worker, daemon=True)
            t.daemon = True
            t.start()
            threads.append(t)

        # Wait until all tasks in queue are finished
        self.queue.join()

        # Stop worker threads
        for _ in range(self.max_threads):
            self.queue.put(None)
        for t in threads:
            t.join()

        return self.visited

if __name__ == '__main__':
    # Example usage
    session = Session()
    load_session(session)  # Load the session with login cookies or headers
    start_url = 'https://www.uni-augsburg.de/de/portal/intranet/'
    crawler = ThreadedCrawler(session, start_url, max_threads=os.cpu_count()) # avoid rate-limiting
    visited_urls = crawler.run()
    print(f'Visited {len(visited_urls)} URLs:')
    for url in visited_urls:
        print(url)

    pages = crawler.pages
    print('saving pages to crawled_pages.json')
    with open('crawled_pages.json', 'w') as f:
        json.dump(pages, f, indent=4)
