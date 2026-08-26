'''
Testing for the login to the University of Augsburg Intranet.
'''

import os
from login.login import login
from requests import Session


session = Session()

login(session=session,
      user_name=os.environ.get('USERNAME'),
      password=os.environ.get('PASSWORD'),
      secret=os.environ.get('TOTP_SECRET'),
      cookie_path=os.environ.get('COOKIE_PATH'))

res = session.get('https://www.uni-augsburg.de/de/portal/intranet/')
