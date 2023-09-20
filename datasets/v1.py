import random

import numpy as np
import pandas as pd
from google.oauth2.service_account import Credentials
from torch.utils.data import Dataset, DataLoader

import directories


def generate_dataset():
    credentials = Credentials.from_service_account_file(directories.bigquery_google_credential)
    with open(directories.queries, "r") as file:
        query = file.read()
    df = pd.read_gbq(
        query,
        project_id="oheadline",
        credentials=credentials,
    )
    print(df.head())
    df.to_csv("bigquery_results_20230920.csv", index=False)


def np_stack_collate(batch):
    # Unzip the batch
    candidate_texts, clicked_texts, browsed_texts = zip(*batch)

    # Stack numpy arrays
    candidate_texts = np.stack(candidate_texts, axis=0)
    clicked_texts = np.stack(clicked_texts, axis=0)
    browsed_texts = np.stack(browsed_texts, axis=0)

    return candidate_texts, clicked_texts, browsed_texts


class OheadlineDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_articles: int = 32,
        sequence_size: int = 20,
        embedding_dim: int = 768,
        K: int = 4,
    ):
        self.dataframe = dataframe
        self.user_ids = self.dataframe["user_id"].unique()
        self.user_data_groups = dict(tuple(self.dataframe.groupby("user_id")))
        self.max_articles = max_articles
        self.sequence_size = sequence_size
        self.embedding_dim = embedding_dim
        self.K = K

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, _):
        user_id = random.choice(self.user_ids)
        user_data = self.user_data_groups[user_id]

        titles = user_data["title"].values.tolist()[:64]
        has_viewed_list = user_data["has_viewed"].values.tolist()[:64]

        assert len(titles) == len(has_viewed_list)

        clicked_texts = [x[0] for x in zip(titles, has_viewed_list) if x[1] is True]
        browsed_texts = [
            x[0]
            for x in zip(titles, has_viewed_list)
            if x[1] is False and x[0] not in clicked_texts
        ]

        if len(clicked_texts) < 2 or len(browsed_texts) < 4:
            return self.__getitem__(_)

        random.shuffle(clicked_texts)
        random.shuffle(browsed_texts)

        candidate_text, clicked_texts = clicked_texts[0], clicked_texts[1 : self.max_articles]
        browsed_texts = browsed_texts[:4]

        # padding for clicked_texts and browsed_texts
        clicked_texts = clicked_texts + [""] * (self.max_articles - len(clicked_texts))
        browsed_texts = (browsed_texts * (4 // len(browsed_texts)))[:4]

        # text, list[32], list[4]
        # Convert to numpy arrays with dtype='object' for strings
        return [candidate_text], clicked_texts, browsed_texts


if __name__ == "__main__":
    dataframe = pd.read_csv(directories.bq_results_csv)
    dataset = OheadlineDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=np_stack_collate)
    for x1, x2, x3 in dataloader:
        print(x1)
        print(x2)
        print(x3)
