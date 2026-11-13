import os

import bcrypt
from dotenv import load_dotenv
from postql import DB
from postql.types import ANSWER_TYPE, Credentials

load_dotenv()

creds = Credentials(
    user=os.environ.get("USERDB"), # type: ignore
    password=os.environ.get("POSTGRES_PASSWORD"), # type: ignore
    host=os.environ.get("POSTGRES_HOST"), # type: ignore
    port=int(os.environ.get("PORT")), # type: ignore
    database_name=os.environ.get("DBNAME") # type: ignore
)

db = DB(creds)


def hash_pwd(password: str) -> str:
    """
    Hashes a password
    Args:
        password (str): the password to hash
    Returns:
        str: the hashed password
    """
    hashed_pwd = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    return hashed_pwd.decode()

def match_pwd(password: str, hashed: str) -> bool:
    """
    Compares a raw and a hashed password
    Args:
        password (str): the password to compare
        hashed (str): the hashed password to compare
    Returns:
        str: whether the passwords match
    """
    return bcrypt.checkpw(password.encode(), hashed.encode())


def get_user(username: str) -> dict | None:
    """
    Gets a user from the database.

    Args:
        username (str): the username of the user
        password (str): the password of the user
    Returns:
        dict | None: the user data if found, None otherwise
    """
    res = db.select(
        table='data.auth_users',
        type_of_answer=ANSWER_TYPE.SINGLE_ANSWER,
        conditions={'username': username.strip().lower()},
        columns=['username', 'password', 'role']
    )
    if res.is_error or not res.data:
        return
    return res.data

def create_user(username: str, password: str, role: str = 'user') -> None | Exception:
    """
    Creates a user in the database.

    Args:
        username (str): the username of the user
        password (str): the password of the user
        role (str): the role of the user
    Raises:
        Exception: if the user could not be created
    Returns:
        None: nothing if the user was created successfully
    """
    hashed_pwd = hash_pwd(password)
    res = db.insert(
        table='data.auth_users',
        values={'username': username.strip().lower(), 'password': hashed_pwd, 'role': role},
    )
    if res.is_error:
        return Exception(f"Error creating user: {res.error}")
    return None
