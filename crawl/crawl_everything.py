from requests import Session
from login.login import load_session
import os
import json
from crawl.crawl_intranet import ThreadedCrawler

# Example usage
session = Session()
load_session(session)  # Load the session with login cookies or headers
start_url = 'https://www.uni-augsburg.de/'
crawler = ThreadedCrawler(session, start_url, max_threads=(os.cpu_count() or 32) * 2, whitelisted_domains=['https://www.uni-augsburg.de', 'https://brand-portal.uni-augsburg.de', 'https://my.corebook.io/uni-augsburg'], timeout=60) # avoid rate-limiting
visited_urls = crawler.run()
print(f'Visited {len(visited_urls)} URLs:')
for url in visited_urls:
    print(url)

pages = crawler.pages
print('saving pages to crawled_pages_all.json')
with open('crawled_pages_all.json', 'w') as f:
    json.dump(pages, f, indent=4)
