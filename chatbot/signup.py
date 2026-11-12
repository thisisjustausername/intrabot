import os
from pathlib import Path

from chainlit.config import config
from chainlit.server import app
from dotenv import load_dotenv
from fastapi import APIRouter, Response
from fastapi.responses import FileResponse
from postql import DB
from postql.types import ANSWER_TYPE, Credentials
from pydantic import BaseModel

from chatbot import auth

load_dotenv()

creds = Credentials(
    user=os.environ.get('USERDB'), # type: ignore
    password=os.environ.get('POSTGRES_PASSWORD'), # type: ignore
    host=os.environ.get('POSTGRES_HOST'), # type: ignore
    port=int(os.environ.get('PORT')), # type: ignore
    database_name=os.environ.get('DBNAME') # type: ignore
)
db = DB(creds)

class SignupBody(BaseModel):
    username: str = ''
    password: str = ''

PAGE = Path(__file__).parent / 'templates/signup.html'
router = APIRouter(prefix=config.run.root_path)

@router.get('/signup')
def signup():
    return FileResponse(PAGE, media_type='text/html', headers={'Cache-Control': 'no-store'})

@router.post('/signup')
def create_user(body: SignupBody):
    username = body.username.lower().strip()
    password = body.password
    role = 'user'

    if not password or not username:
        return Response('Username and password are required', status_code=400)

    res = db.select(
        table='data.auth_users',
        type_of_answer=ANSWER_TYPE.SINGLE_ANSWER,
        conditions={'username': username},
        columns=['username']
    )

    if res.is_error:
        return Response('Error checking username', status_code=500)

    if res.data:
        return Response('Username already exists', status_code=409)

    res = auth.create_user(username=username, password=password, role=role)
    if isinstance(res, Exception):
        return Response('Error creating user', status_code=500)

    return Response('User created successfully', status_code=201)

_before = len(app.router.routes)
app.include_router(router)
_ours = app.router.routes[_before:]
del app.router.routes[_before:]
app.router.routes[0:0] = _ours
