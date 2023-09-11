import os

from pathlib import Path

project = Path(os.path.dirname(__file__)).parent

backends = project.joinpath("backends")

http = backends.joinpath("http")

static = http.joinpath("static")

logging = project.joinpath("logging.yaml")

sqlite3 = project.joinpath("sqlite3.db")

credentials = project.joinpath("credentials")

bigquery_google_credential = credentials.joinpath("bigquery-google-credential.json")

bq_results = project.joinpath("bq-results-20230901.csv")
