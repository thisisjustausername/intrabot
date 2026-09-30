# intrabot
Semantically index the Uni Augsburg Intranet and create Chatbot for interaction.<br/>
Build semantic search engine with chatbot for intranet of University of Augsburg.

# Workflow
Follow this workflow to successfully host your private intrabot instance.<br/>
We advise to use a powerful NVIDIA-GPU for this workflow.

1. Start the basic setup
```bash
git clone https://github.com/thisisjustausername/intrabot
cd intrabot

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

touch login/cookies.pkl
```

2. Manually insert your login information in your .env file
We provide a dummy .env file:
```ini
USERNAME=my-rz-kennung
PASSWORD='my-rz-password'
TOTP_SECRET='MY TOTP SECRET'
COOKIE_PATH=base_path_to_project_parent_folder/intrabot/login/cookies.pkl
```

3. Run pipeline
```bash
python3 -m login.login
python3 -m crawl.crawl_intranet
python3 -m add_search.create_embeddings
python3 -m add_search.add_lexical_search
```

4. Finally run your intrabot chatbot with the first command or intraSearch with the second command
```bash
chainlit run chatbot/graph.py --host 127.0.0.1 --port 8001
```
```bash
python3 -m add_search.raw_semantic_search
```

# IMPORTANT TODOS
* FILTER OUT WEBPAGES AFTER CRAWLING: https://collab.dvb.bayern/users/, ...
* add login for collab bayern as it does not allow totp
* in create_embeddings.py: optimize memory: currently saving full doc for each chunk: a lot of redundant data
