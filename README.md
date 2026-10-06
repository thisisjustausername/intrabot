# intrabot
Semantically index the Uni Augsburg Intranet and create Chatbot for interaction.<br/>
Build semantic search engine with chatbot for intranet of University of Augsburg.

# Workflow
Follow this workflow to successfully host your private intrabot instance.<br/>
We advise to use a powerful NVIDIA-GPU for this workflow.

1. Start the basic setup
    ```bash
    git clone https://github.com/thisisjustausername/intrabot
    git clone https://github.com/thisisjustausername/mhbai
    cd intrabot
    
    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    pip3 install -e ../mhbai
    
    touch login/cookies.pkl
    ```

2. Manually insert your login information in your .env file<br/>
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
    python3 -m search.create_embeddings
    python3 -m search.add_lexical_search
    ```

4. Start LLM-Backend with vLLM<br/>
  For `tensor-parallel-size` specify the amount of individual GPUs you plan on using.
    ```bash
    python -m vllm.entrypoints.openai.api_server --model nvidia/Qwen3.8-27B-NVFP4 --max-model-len 131072 --port 8000 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --tensor-parallel-size 2
    ```
    In case you do not want to restart vLLM on every bootup, set a systemd-service as followed.
    ```ini
    [Unit]
    Description=Running vllm for intrabot intranet chatbot from intrabot project by Leon Gattermeyer
    After=network.target
    
    [Service]
    Type=simple
    User=<USERNAME>
    WorkingDirectory=<WORKING DIRECTORY OF INTRABOT PROJECT>
    Environment="PATH=<WORKING DIRECTORY OF INTRABOT PROJECT>/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    Environment="VIRTUAL_ENV=<WORKING DIRECTORY OF INTRABOT PROJECT>/venv"
    Environment="CUDA_HOME=/usr/local/cuda"
    ExecStart=<WORKING DIRECTORY OF INTRABOT PROJECT>/venv/bin/python3 -m vllm.entrypoints.openai.api_server --model nvidia/Qwen3.8-27B-NVFP4 --max-model-len 131072 --port 8000 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --tensor-parallel-size 2
    Restart=always
    
    [Install]
    WantedBy=multi-user.target
    ```
    Then simply enable and start the service.
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable vllm.service
    sudo systemctl start vllm.service
    ```

5. Set up the persistent storage and login to the chatbot in Postgres.<br/>
    ```bash
    sudo -u postgres psql
    CREATE USER <USERNAME>;
    ALTER ROLE  <USERNAME> SET search_path = bot, auth;
    CREATE DATABASE chlit OWNER <USERNAME>;
    GRANT ALL PRIVILEGES ON DATABASE chlit TO <USERNAME>;
    exit
    psql -d chlit -f chatbot/data_storage/schema.sql
    ```
    


6. Finally run your intrabot chatbot with the first command or intraSearch with the second command
    ```bash
    chainlit run chatbot/graph.py --host 127.0.0.1 --port 8001
    ```
    ```bash
    python3 -m search.raw_semantic_search
    ```

# IMPORTANT TODOS
* FILTER OUT WEBPAGES AFTER CRAWLING: https://collab.dvb.bayern/users/, ...
* add login for collab bayern as it does not allow totp
* in create_embeddings.py: optimize memory: currently saving full doc for each chunk: a lot of redundant data
