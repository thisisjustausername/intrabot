import os

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

def search_db(url: str) -> str | Exception:
    res = db.select(
        table='data.website',
        type_of_answer=ANSWER_TYPE.SINGLE_ANSWER,
        columns=['markdown'],
        conditions={'url': url}
    )
    if res.is_error:
        print(f"Error fetching data from database: {res.error}")
        raise Exception(f"Error fetching data from database")
    data = res.data
    if data and len(data) > 0:
        return data['markdown']
    return 'No content found for this url, this url might be invalid'

if __name__ == "__main__":
    existing_url = 'https://www.uni-augsburg.de/de/'
    non_existing_url = 'https://www.nonexistenturl.com'
    print(search_db(existing_url))
    print(search_db(non_existing_url))
