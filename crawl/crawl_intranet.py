'''
This script implements a threaded web crawler that starts from a given URL and crawls all reachable pages within the same domain. It uses the `requests` library to make HTTP requests, `BeautifulSoup` to parse HTML, and `html_to_markdown` to convert HTML content to Markdown format. The crawler is designed to handle multiple threads for concurrent crawling, improving efficiency.

Start the backend using:
```bash
NUM_GPUS=2
PER_GPU=15
for ((i=0; i<NUM_GPUS*PER_GPU; i++)); do
    p=$((5001 + i))
    gpu=$(( (p - 5001) % NUM_GPUS ))   # round-robin ports across GPUs
    CUDA_VISIBLE_DEVICES=$gpu \
    DOCLING_DEVICE=cuda \
    DOCLING_NUM_THREADS=8 DOCLING_PERF_PAGE_BATCH_SIZE=4 \
    DOCLING_SERVE_ENG_LOC_NUM_WORKERS=1 \
    docling-serve run --port $p &
done
```
'''
import json
import os
import re
import threading
from io import BytesIO
from queue import Queue
from urllib.parse import urljoin, urlparse

import html_to_markdown as htm
import requests
from bs4 import BeautifulSoup
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from dotenv import load_dotenv
from postql import DB
from postql.types import Credentials
from requests import Session

from login.login import load_session

load_dotenv()

opts = PdfPipelineOptions()
opts.do_ocr = False
opts.do_table_structure = True
opts.table_structure_options.mode = TableFormerMode.FAST # type: ignore
opts.accelerator_options = AcceleratorOptions(num_threads=32, device=AcceleratorDevice.CUDA)

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
)

NUM_GPUS = 2
PER_GPU = 15
DOCLING_SERVERS = [f"http://localhost:{p}" for p in (5001 + i for i in range(NUM_GPUS * PER_GPU))]

page_break = "---Page Break---"

creds = Credentials(
    user=os.environ.get("USERDB"), # type: ignore
    password=os.environ.get("POSTGRES_PASSWORD"), # type: ignore
    host=os.environ.get("POSTGRES_HOST"), # type: ignore
    port=int(os.environ.get("PORT")), # type: ignore
    database_name=os.environ.get("DBNAME") # type: ignore
)

db = DB(creds)

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
    def __init__(self,
        session: Session,
        start_urls: list[str],
        max_threads: int = 5,
        whitelisted_domains: set[str] | None = None,
        blacklisted_domains: set[str] | None = None,
        timeout: int = 30,
        visited: set[str] | None = None):

        self.start_urls = start_urls
        self.pdf_queue = Queue(maxsize=NUM_GPUS * PER_GPU * 10)
        self.parser_servers = DOCLING_SERVERS
        self.num_parsers = len(DOCLING_SERVERS)
        self.queue = Queue()
        self.visited = set() if visited is None else visited
        self.lock = threading.Lock()
        self.session = session
        self.max_threads = max_threads
        self.options = htm.ConversionOptions(exclude_selectors=['script', 'style', 'noscript', 'footer', 'nav'])
        self.whitelisted_domains = whitelisted_domains if whitelisted_domains is not None else {r'^https://www\.([a-zA-Z0-9-]+\.)?uni-augsburg\.de(/.*)?$', r'^https://brand-portal\.uni-augsburg\.de(/.*)?$', r'^https://my\.corebook\.io/uni-augsburg(/.*)?$'}
        self.blacklisted_domains = blacklisted_domains if blacklisted_domains is not None else {r'^https://collab.dvb.bayern/users(/.*)?$', r'^https://www\.([a-zA-Z0-9-]+\.)?uni-augsburg\.de/en(/.*)?$'}
        self.timeout = timeout
        # self.pages: list[tuple[str, str]] = []
        self.error_urls = set()

    def normalize_url(self, base_url, href):
        '''Converts relative URLs to absolute and strips fragments (#anchor).'''
        if not href.startswith('https://'):
            href = urljoin(base_url, href)
        parsed = urlparse(href)
        clean_url = f'{parsed.scheme}://{parsed.netloc}{parsed.path}'.rstrip('/')
        if parsed.query:
            clean_url += f'?{parsed.query}'
        return clean_url

    def pdf_to_markdown(self, data: bytes) -> str:
        '''
        Convert PDF bytes to Markdown format.

        Args:
            data (bytes): The PDF file content in bytes.

        Returns:
            str: The converted Markdown content.
        '''
        stream = DocumentStream(name="document.pdf", stream=BytesIO(data))
        result = converter.convert(stream)
        md_doc = result.document.export_to_markdown(page_break_placeholder=page_break)
        return md_doc

    def convert_pdf(self, data: bytes, server_url) -> str:
        stream = BytesIO(data)
        response = requests.post(
            f"{server_url}/v1/convert/file",
            files={
                "files": (
                    'document.pdf',
                    stream,
                    "application/pdf",
                )
            },
            data={
                "from_formats": "pdf",
                "to_formats": "md",
                "do_ocr": "false",
                "do_table_structure": "true",
                "table_mode": "fast",
            },
            timeout=600,
        )

        response.raise_for_status()
        result = response.json()

        # Extract Markdown
        md_doc = result["document"]["md_content"]

        # Add your preferred page-break placeholder
        md_doc = md_doc.replace("\n---\n", f"\n{page_break}\n")
        return md_doc

    def parser(self, jobs, server):
        while True:
            item = jobs.get()
            if item is None:
                return

            try:
                url, file = item
                md = self.convert_pdf(file, server)
                res = db.insert(
                    table="data.website",
                    values={
                        "url": url,
                        "type": "pdf",
                        "markdown": md
                    }
                )

                if res.is_error:
                    print(f"\033[31mFailed to insert {url} into database: {res.error}\033[0m")
            except Exception as e:
                print(f"Error processing {url}: {e}")


    def worker(self):
        while True:
            url = None
            try:
                url = self.queue.get()
                if url is None:
                    return
                response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
                content_type = response.headers.get('Content-Type', '')
                page_url = response.url or url  # Use the final URL after redirects if available
                if response.status_code != 200:
                    continue
                if content_type == 'application/pdf':
                    MAX_BYTES = 100 * 1024 * 1024  # 100 MB
                    # Stream the body with a size limit
                    chunks, size = [], 0
                    for chunk in response.iter_content(64 * 1024):
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise ValueError(f"PDF too large: {url}")
                        chunks.append(chunk)
                    data = b"".join(chunks)

                    if not data.startswith(b"%PDF-"):
                        raise ValueError(f"Claimed PDF but bad magic bytes: {url}")
                    self.pdf_queue.put((page_url, data))
                    if page_url not in self.visited:
                        with self.lock:
                            self.visited.add(page_url)
                    # continue here allows to avoid waiting for PDF to MD conversion to finish
                    print(url, page_url)
                    continue

                if 'text/html' not in content_type:
                    continue
                markdown = htm.convert(response.text, options=self.options)
                if page_url not in self.visited:
                    with self.lock:
                        self.visited.add(page_url)
                # TODO: search for urls in pdfs too
                res = db.insert(
                    table="data.website",
                    values={
                        "url": page_url,
                        "type": "pdf",
                        "markdown": markdown.content
                    }
                )

                if res.is_error:
                    print(f"\033[31mFailed to insert {url} into database: {res.error}\033[0m")
                # self.pages.append((page_url, markdown.content if skip_md is False else markdown))
                print(url, page_url)

                soup = BeautifulSoup(response.text, 'html.parser')
                base_tag = soup.find('base', href=True)
                if base_tag:
                    # If base_tag is itself relative, resolve it against page_url
                    base_url = urljoin(page_url, base_tag['href']) # type: ignore
                else:
                    base_url = page_url

                for a_tag in soup.find_all('a', href=True):
                    link = self.normalize_url(base_url, a_tag['href'])
                    if link is None:
                        continue
                    ll = link.strip().lower()
                    if not any(re.match(i, ll) for i in self.whitelisted_domains): continue
                    if any(re.match(i, ll) for i in self.blacklisted_domains): continue
                    with self.lock:
                        if link not in self.visited:
                            self.visited.add(link)
                            self.queue.put(link)
            except Exception as e:
                print(f'\033[31mError crawling {url}: {e}\033[0m')
                self.error_urls.add(url)
            finally:
                self.queue.task_done()

    def run(self):
        # Seed initial URL
        for start_url in self.start_urls:
            self.visited.add(start_url)
            self.queue.put(start_url)

        # Start parser threads
        parser_threads = []
        for server in self.parser_servers:
            t = threading.Thread(target=self.parser, args=(self.pdf_queue, server))
            t.start()
            parser_threads.append(t)

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

        for _ in parser_threads:
            self.pdf_queue.put(None)
        for t in parser_threads:
            t.join()

        return self.visited

if __name__ == '__main__':
    # Example usage
    session = Session()
    load_session(session)  # Load the session with login cookies or headers
    start_url = 'https://www.uni-augsburg.de/de/portal/intranet/'
    crawler = ThreadedCrawler(
        session,
        [start_url],
        max_threads=(os.cpu_count() or 32) * 10,
        whitelisted_domains={r'^https://www\.([a-zA-Z0-9-]+\.)?uni-augsburg\.de(/.*)?$', r'^https://brand-portal\.uni-augsburg\.de(/.*)?$', r'^https://my\.corebook\.io/uni-augsburg(/.*)?$'}, # , r'^https://collab\.dv\b.bayern(/.*)?$'
        blacklisted_domains={r'^https://collab.dvb.bayern/users(/.*)?$', r'^https://www\.([a-zA-Z0-9-]+\.)?uni-augsburg\.de/en(/.*)?$', r'^https://collab\.dvb\.bayern/login.action\?.*?$', r'^https://www\.uni-augsburg\.de/admin/login([/?].*)?$', r'^https://collab.dvb.bayern(/.*)?$', r'^https://www.uni-augsburg.de/admin/logout([/?].*)?', r'^https://www.uni-augsburg.de/auth([/?].*)?'},
        timeout=60,
        visited=None
    )
    visited_urls = crawler.run()
    print(f'Visited {len(visited_urls)} URLs:')
    for url in visited_urls:
        print(url)

    # pages = crawler.pages
    print('saving pages to crawled_pages.json')
    # with open('crawled_pages.json', 'w') as f:
    #     json.dump(pages, f, indent=4)
    with open('stats_crawled.json', 'w') as f:
        json.dump({
            'visited_urls': list(visited_urls),
            'error_urls': list(crawler.error_urls),
      #       'pages_count': len(pages)
        }, f, indent=4)
