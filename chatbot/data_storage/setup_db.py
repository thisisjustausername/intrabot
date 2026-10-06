# Copyright (c) 2025 Leon Gattermeyer
#
# This file is part of mhbai.
#
# Licensed under the AGPL-3.0 License. See LICENSE file in the project root for full license information.

# Description: setup database connection parameters
# Status: VERSION 1.0
# FileID: Sc-da-0005

import getpass
import os

from dotenv import load_dotenv, set_key

# Set path for .env file
# NOTE: very dirty
env_path = os.getcwd().split('/chatbot', 1)[0] + '/.env'

# Load environment variables from .env file
load_dotenv()
USERDB = getpass.getuser()
HOST = "localhost"
PORT = "5432"
DBNAME = "chlit"
PASSWORD = input(f"Enter the postgresql database password for user {USERDB}: ")
set_key(env_path, "USERDB", USERDB)
set_key(env_path, "HOST", HOST)
set_key(env_path, "PORT", PORT)
set_key(env_path, "DBNAME", DBNAME)
set_key(env_path, "PASSWORD", PASSWORD)
print("Keys saved to .env file.")
