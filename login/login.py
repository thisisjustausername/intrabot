# Copyright (c) 2025 Leon Gattermeyer
'''
This script implements a threaded web crawler that starts from a given URL and crawls all reachable pages within the same domain. It uses the `requests` library to make HTTP requests, `BeautifulSoup` to parse HTML, and `html_to_markdown` to convert HTML content to Markdown format. The crawler is designed to handle multiple threads for concurrent crawling, improving efficiency.
'''


import base64
import hashlib
import hmac
import os
import pickle
import struct
import time
import xml.etree.ElementTree as ET
import zlib
from urllib import parse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


def decode_saml_request(saml_request: str) -> str:
    '''
    Decodes a SAML request in order to extract the XML content.

    Args:
        saml_request (str): The SAML request string
    Returns:
        str: The decoded XML content of the SAML request
    '''
    saml_request = parse.unquote(saml_request)
    decoded = base64.b64decode(saml_request)
    try:
        # Try to decompress (some SAMLRequests are compressed using DEFLATE)
        inflated = zlib.decompress(decoded, -15)  # -15 to skip zlib header
        return inflated.decode()
    except zlib.error:
        # If not compressed, just return base64-decoded value
        return decoded.decode()


def generate_totp(secret: str, digits: int=6, interval: int=30, digest=hashlib.sha1) -> str:
    '''
    Generates a TOTP token based on the provided secret.

    Args:
        secret (str): The base32-encoded secret used to generate the TOTP token
        digits (int): The number of digits in the TOTP token (default is 6)
        interval (int): The time interval in seconds for TOTP (default is 30 seconds)
        digest (callable): The hash function to use (default is hashlib.sha1)
    Returns:
        str: The generated TOTP token as a string of digits
    '''
    secret = secret.replace(' ', '')
    # Decode base32 secret to bytes
    key = base64.b32decode(secret.upper())

    # Get current Unix time and divide by interval
    counter = int(time.time() // interval)

    # Pack counter into 8-byte big-endian
    msg = struct.pack('>Q', counter)

    # Create HMAC-SHA1 digest
    hmac_digest = hmac.new(key, msg, digest).digest()

    # Dynamic truncation (RFC 4226)
    offset = hmac_digest[-1] & 0x0F
    code = struct.unpack('>I', hmac_digest[offset:offset + 4])[0] & 0x7FFFFFFF

    # Return last <digits> digits as TOTP token
    return str(code % (10 ** digits)).zfill(digits)


def login(session: requests.Session, user_name: str, password: str, secret: str, cookie_path: str) -> tuple[requests.Response, requests.Session]:
    '''
    Logs into the Digicampus using TOTP authentication.
    If save_sess is True, the session will be saved to a file.

    Args:
        session (requests.Session): The requests session object to use for the login process
        user_name (str): The username for the Digicampus login
        password (str): The password for the Digicampus login
        secret (str): The base32-encoded secret used to generate the TOTP token
        cookie_path (str): The path to save the session cookies if save_sess is True
    Returns:
        tuple[requests.Response, requests.Session]: A tuple containing the response from the final login request and the session object.
    '''

    # url to start the login process
    start_url = 'https://www.uni-augsburg.de/de/portal/intranet/'
    result = session.get(start_url)

    # extract the login URL from the page
    soup = BeautifulSoup(result.text, 'lxml')
    next_step_url = soup.find('form', id='kc-form-login').get('action')

    # pass in user data to get to the next step
    result = session.post(next_step_url, data={'username': user_name, 'password': password}) # type: ignore
    soup = BeautifulSoup(result.text, 'lxml')
    field = soup.find('form', id='kc-otp-login-form')
    selectedCredentialId = field.find('input', id='selectedCredentialId').get('value') # type: ignore
    final_url = field.get('action') # type: ignore

    # send the TOTP token to complete the login process
    result = session.post(final_url, data={'selectedCredentialId': selectedCredentialId, 'otp': generate_totp(secret)}) # type: ignore

    save_session(file=cookie_path, session=session)

    return result, session


@NotImplementedError
def login_collab_dvb_bayern(session: requests.Session, user_name: str, password: str, secret: str, cookie_path: str) -> tuple[requests.Response, requests.Session]:
    '''
    Logs into the Collab DVB Bayern using TOTP authentication.
    Args:
        session (requests.Session): The requests session object to use for the login process
        user_name (str): The username for the Collab DVB Bayern login
        password (str): The password for the Collab DVB Bayern login
        secret (str): The base32-encoded secret used to generate the TOTP token
        cookie_path (str): The path to save the session cookies if save_sess is True
    '''

    # url to start the login process
    start_url = 'https://collab.dvb.bayern/plugins/servlet/samlsso?redirectTo=%2F'
    result = session.get(start_url)
    soup = BeautifulSoup(result.text, 'lxml')

def save_session(file: str, session: requests.Session) -> None:
    """
    Saves a session to a file.

    Args:
        file (str): The path to the file where the session cookies will be saved.
        session (requests.Session): The session to be saved.
    """
    with open(file, "wb") as pkl:
        pickle.dump(session.cookies, pkl)


def load_session(session: requests.Session) -> requests.Session:
    """
    Loads a session from a file.
    It is advised to use def load_session instead of the slower def login.

    Args:
        file (str): The path to the file containing the session cookies.
        session (requests.Session): The session to which the cookies will be loaded.
    Returns:
        requests.Session: The session loaded from the file.
    """

    # printing login information
    print("Logging in...")

    # load login parameters from .env
    load_dotenv()
    user_name: str = os.getenv('USERNAME')
    password: str = os.getenv('PASSWORD')
    secret: str = os.getenv('TOTP')
    cookie_path: str = os.getenv('COOKIE_PATH')

    # reading the saved cookies
    try:
        with open(cookie_path, "rb") as pkl:
            cookies = pickle.load(pkl)
    except EOFError:
        print("No cookies saved yet. Logging in.")
        login(session=session, user_name=user_name, password=password, secret=totp_secret, cookie_path=cookie_path)
        return session
    session.cookies = cookies
    if "login-pf" in session.get("https://www.uni-augsburg.de/de/portal/intranet/").text:
        print("Session expired. Logging in.")
        login(session=session, user_name=user_name, password=password, secret=totp_secret, cookie_path=cookie_path)
        return session
    return session

if __name__ == '__main__':
    load_dotenv()
    user_name = os.getenv('USERNAME')
    password = os.getenv('PASSWORD')
    secret = os.getenv('TOTP_SECRET')
    cookie_path = os.getenv('COOKIE_PATH')
    session = requests.Session()
    login(session=session, user_name=user_name, password=password, secret=secret, cookie_path=cookie_path)
    res = session.get("https://www.uni-augsburg.de/de/portal/intranet/")
    print(res.text)
    print(res.status_code)
