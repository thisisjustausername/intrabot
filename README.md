# intrabot
Semantically index the Uni Augsburg Intranet and create Chatbot for interaction.<br/>
Build semantic search engine with chatbot for intranet of University of Augsburg.

# IMPORTANT NOTICE
Highest order TODOs
* Set maximum amount of tool calls
* Fix search result order for mongodb (search informatik e.g.)
* Nemotron uncertainty confidence besser angeben

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
    # Download .tar file
    curl https://raw.githubusercontent.com/thisisjustausername/postql/refs/heads/main/dist/postql-0.1.0.tar.gz --output postql-0.1.0.tar.gz
    
    # Install .tar file
    pip install postql-0.1.0.tar.gz
    
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

3. Set up the persistent storage and login to the chatbot in Postgres as well as the persistent storage for the crawling.<br/>
    ```bash
    sudo -u postgres psql
    CREATE USER <USERNAME>;
    ALTER ROLE  <USERNAME> SET search_path = bot, auth, data;
    CREATE DATABASE chlit OWNER <USERNAME>;
    GRANT ALL PRIVILEGES ON DATABASE chlit TO <USERNAME>;
    exit
    psql -d chlit -f chatbot/data_storage/schema.sql
    ```

4. Run pipeline
    This only has to be run once for setting up a TOTP-token for the 2FA-login to the intrant:
    * Follow the instructions for adding a TOTP-token and instead of scanning the QR-code, copy the TOTP-secret and paste it into your .env file.
    * Then run following:
      ```python
      from login.login import generate_totp

      # insert your TOTP-SECRET here
      print(generate_totp(<TOTP_SECRET>))
      ```
    * Paste the generated TOTP-token into the regarding field on the website to finish TOTP-setup.
    ---
    Now you can run the data crawling and indexing pipeline any time you want.

    For crawling you need to start two separate session:
    * In the first session run the parser that convert PDFs to Markdown documents.
      ```bash
      screen

      # Enter

      cd <WORKING DIRECTORY OF INTRABOT PROJECT>/intrabot
      source venv/bin/activate

      # this can be found in the module docstring of the crawl/crawl_intrabot.py file.
      # Match the params NUM_GPUS, PER_GPU and DOCLING_SERVE_MAX_SYNC_WAIT (with the timeout parameter) to the code in that file.
      NUM_GPUS=2
      PER_GPU=10
      for ((i=0; i<NUM_GPUS*PER_GPU; i++)); do
          p=$((5001 + i))
          gpu=$(( (p - 5001) % NUM_GPUS ))   # round-robin ports across GPUs
          CUDA_VISIBLE_DEVICES=$gpu \
          DOCLING_DEVICE=cuda \
          DOCLING_NUM_THREADS=8 DOCLING_PERF_PAGE_BATCH_SIZE=4 \
          DOCLING_SERVE_ENG_LOC_NUM_WORKERS=1 \
          DOCLING_SERVE_MAX_SYNC_WAIT=600 \
          docling-serve run --port $p &
      done

      # detach from the session
      <Ctrl-A-D>
      ```

      Then crawl the websites.
      ```bash
      python3 -m crawl.crawl_everything
      ```

    Finally create the embeddings and add the lexical search to the database.\
    ```bash
    python3 -m search.create_lexical_search
    ```

5. Start LLM-Backend with vLLM<br/>
  For `tensor-parallel-size` specify the amount of individual GPUs you plan on using.
  Note, that nemotron-3.5-super has a max model length of 1_000_000 tokens.
    ```bash
    vllm serve <WORKING DIRECTORY OF INTRABOT PROJECT>/model/nemotron-120B \
      --served-model-name nemotron-3.5-super \
      --trust-remote-code \
      --tensor-parallel-size 2 \
      --max-model-len 262144 \
      --port 8000 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --reasoning-parser nemotron_v3 \
    --limit-mm-per-prompt '{"image":2,"video":0}'
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
    ExecStart=<WORKING DIRECTORY OF INTRABOT PROJECT>/venv/bin/vllm serve <WORKING DIRECTORY OF INTRABOT PROJECT>/model/nemotron-120B --served-model-name nemotron-3.5-super --trust-remote-code --tensor-parallel-size 2 --max-model-len 262144 --port 8000 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --limit-mm-per-prompt '{"image":2,"video":0}'
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


6. Finally run your intrabot chatbot with the first command or intraSearch with the second command
    ```bash
    python3 -m chainlit run chatbot/graph.py --host 127.0.0.1 --port 8001 --root-path /intrabot
    ```
    ```bash
    python3 -m search.raw_semantic_search
    ```

# IMPORTANT TODOS
* Split XML MHBS into modules exams etc as usual but keep the xml, then convert xml to md and for each document (same splitting as in current version) save md card and maybe infocard plus optionally use chunking, then create own vector db or together with intranet db but add filter options like classes of source.
* Set maximum amount of tool calls
* Fix search result order for mongodb (search informatik e.g.)
* FILTER OUT WEBPAGES AFTER CRAWLING: https://collab.dvb.bayern/users/, ...
* add login for collab bayern as it does not allow totp
* in create_embeddings.py: optimize memory: currently saving full doc for each chunk: a lot of redundant data
* create src/ package folder structure
ture
