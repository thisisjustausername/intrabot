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
python3 -m add_semantic_search.create_embeddings
```

4. Finally run your intrabot chatbot with the first command or intraSearch with the second command
```bash
chainlit run add_semantic_search/graph.py --host 127.0.0.1 --port 8001
```
```bash
python3 -m add_semantic_search.raw_semantic_search
```
