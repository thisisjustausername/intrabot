import json
import os

from requests import Session

from crawl.crawl_intranet import ThreadedCrawler
from login.login import load_session

# TODO: for redirects use last url for saving

# Example usage
session = Session()
load_session(session)  # Load the session with login cookies or headers
start_urls = ['https://www.uni-augsburg.de', 'https://www.uni-augsburg.de/de/portal/intranet/', 'https://collab.dvb.bayern/', 'https://collab.dvb.bayern/spaces/UniARZSER/pages/395639043/Knowledge+Base+des+Rechenzentrums', 'https://collab.dvb.bayern/spaces/UniAZV2/pages/751046293/Knowledge+Base+der+Personalabteilung+der+Universit%C3%A4t+Augsburg', 'https://collab.dvb.bayern/spaces/UniARZHPCKB/pages/392035423/Knowledge+Base+f%C3%BCr+wissenschaftliches+Rechnen+HPC+Startseite']

stats = None
try:
    with open('stats_crawled_all.json', 'r') as f:
        stats = json.load(f)
except FileNotFoundError:
    pass

previously_visited_urls = set(stats.get('visited_urls', [])) if stats else None
previously_error_urls = set(stats.get('error_urls', [])) if stats else None

crawler = ThreadedCrawler(
    session,
    start_urls,
    max_threads=(os.cpu_count() or 32) * 10,
    whitelisted_domains={r'^https://www\.([a-zA-Z0-9-]+\.)?uni-augsburg\.de(/.*)?$', r'^https://brand-portal\.uni-augsburg\.de(/.*)?$', r'^https://my\.corebook\.io/uni-augsburg(/.*)?$', r'^https://collab\.dv\b.bayern(/.*)?$', r'^https://www\.uni-augsburg\.de/admin/login(/.*)?$'},
    blacklisted_domains={r'^https://collab.dvb.bayern/users(/.*)?$', r'^https://www\.([a-zA-Z0-9-]+\.)?uni-augsburg\.de/en(/.*)?$', r'^https://collab\.dvb\.bayern/login.action\?.*?$', r'^https://www\.uni-augsburg\.de/admin/login(/.*)?$'},
    timeout=60,
    visited=previously_visited_urls.union(previously_error_urls) if previously_visited_urls is not None else None # type: ignore
)
del previously_visited_urls
for url in previously_error_urls or []:
    crawler.queue.put(url)

visited_urls = crawler.run()
print(f'Visited {len(visited_urls)} URLs:')
for url in visited_urls:
    print(url)

pages = crawler.pages
print('saving pages to crawled_pages_all.json')
try:
    with open('crawled_pages_all.json', 'r') as f:
        existing_pages = json.load(f)
        pages.extend(existing_pages)
except FileNotFoundError:
    pass
with open('crawled_pages_all.json', 'w') as f:
    json.dump(pages, f, indent=4)
with open('stats_crawled_all.json', 'w') as f:
    json.dump({
        'visited_urls': list(visited_urls),
        'error_urls': list(crawler.error_urls),
        'pages_count': len(pages)
    }, f, indent=4)
