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

queries = project.joinpath("queries.sql")

tests = project.joinpath("tests")

csv = project.joinpath("csv")

train_dataset_csv = csv.joinpath("train_dataset_20230920.csv")

unittest_dataset_csv = csv.joinpath("test_dataset_20230920.csv")
