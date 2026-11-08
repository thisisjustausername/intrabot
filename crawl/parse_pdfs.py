"""
Multiprocess PDF to Markdown conversion using Docling API.

Start the backend using:
```bash
NUM_GPUS=2
PER_GPU=10
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
"""

import os
import queue
import threading
import time

import requests
from dotenv import load_dotenv
from postql import DB
from postql.types import Credentials

load_dotenv()

creds = Credentials(
    user=os.environ.get("USERDB"), # type: ignore
    password=os.environ.get("POSTGRES_PASSWORD"), # type: ignore
    host=os.environ.get("POSTGRES_HOST"), # type: ignore
    port=int(os.environ.get("PORT")), # type: ignore
    database_name=os.environ.get("DBNAME") # type: ignore
)

db = DB(creds)

NUM_GPUS = 2
PER_GPU = 10

DOCLING_SERVERS = [f"http://localhost:{p}" for p in (5001 + i for i in range(NUM_GPUS * PER_GPU))]
page_break = "---Page Break---"

def convert_pdf(path, server_url):
    with open(path, "rb") as pdf_file:
        response = requests.post(
            f"{server_url}/v1/convert/file",
            files={
                "files": (
                    os.path.basename(path),
                    pdf_file,
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

    res = db.insert(
        table="data.website",
        values={
            "url": path,
            "type": "pdf",
            "markdown": md_doc,
        }
    )
    if res.is_error:
        print(f"Error inserting into database for {path}: {res.error}")
    return md_doc


def run_all(files):
    jobs = queue.Queue()
    for f in files:
        jobs.put(f)

    results, failures = {}, {}

    def worker(server):
        while True:
            try:
                path = jobs.get_nowait()
            except queue.Empty:
                return
            print(f"Starting: {path} on {server}")
            convert_pdf(path, server)



    threads = [threading.Thread(target=worker, args=(s,)) for s in DOCLING_SERVERS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, failures

if __name__ == "__main__":
    files = ['test_pdfs/' + i for i in os.listdir("test_pdfs")]
    start = time.time()
    run_all(files)
    print(f"Total time: {time.time() - start:.2f} seconds")
