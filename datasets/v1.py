import random

import pandas as pd
import torch
from google.oauth2.service_account import Credentials
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer

import directories
import models.v1


def generate_dataset():
    credentials = Credentials.from_service_account_file(directories.bigquery_google_credential)
    with open(directories.queries, "r") as file:
        query = file.read()
    df = pd.read_gbq(
        query,
        project_id="oheadline",
        credentials=credentials,
    )

    def only_active_users(group):
        has_viewed_true_count = group[group["has_viewed"] == True].shape[0]
        has_viewed_false_count = group[group["has_viewed"] == False].shape[0]
        return has_viewed_true_count > 2 and has_viewed_false_count > 4

    df = df.groupby("user_id").filter(only_active_users)
    df.to_csv("bigquery_results_20230920.csv", index=False)


def batch_encoding_collate(batch) -> tuple[dict, ...]:
    candidate_tokens, clicked_tokens, browsed_tokens = zip(*batch)

    candidate_input_ids = torch.stack([x["input_ids"] for x in candidate_tokens])
    candidate_attention_mask = torch.stack([x["attention_mask"] for x in candidate_tokens])
    _candidate_tokens = {
        "input_ids": candidate_input_ids,
        "attention_mask": candidate_attention_mask,
    }

    clicked_input_ids = torch.stack([x["input_ids"] for x in clicked_tokens])
    clicked_attention_mask = torch.stack([x["attention_mask"] for x in clicked_tokens])
    _clicked_tokens = {
        "input_ids": clicked_input_ids,
        "attention_mask": clicked_attention_mask,
    }

    browsed_input_ids = torch.stack([x["input_ids"] for x in browsed_tokens])
    browsed_attention_mask = torch.stack([x["attention_mask"] for x in browsed_tokens])
    _browsed_tokens = {
        "input_ids": browsed_input_ids,
        "attention_mask": browsed_attention_mask,
    }

    # CANDIDATE_TOKENS
    # input_ids: Tensor: (64, 1, 20)
    # attention_mask: Tensor: (64, 1, 20)
    # CLICKED_TOKENS
    # input_ids: Tensor: (64, 32, 20)
    # attention_mask: Tensor: (64, 32, 20)
    # BROWSED_TOKENS
    # input_ids: Tensor: (64, 4, 20)
    # attention_mask: Tensor: (64, 4, 20)
    return _candidate_tokens, _clicked_tokens, _browsed_tokens


class OheadlineDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_articles: int = 32,
        sequence_size: int = 20,
        embedding_dim: int = 768,
        K: int = 4,
        tokenizer=None,
    ):
        self.dataframe = dataframe
        self.user_ids = self.dataframe["user_id"].unique()
        self.user_data_groups = dict(tuple(self.dataframe.groupby("user_id")))
        self.max_articles = max_articles
        self.sequence_size = sequence_size
        self.embedding_dim = embedding_dim
        self.K = K
        self.tokenizer = tokenizer or ElectraTokenizer.from_pretrained(
            "monologg/koelectra-base-v3-discriminator"
        )

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index: int):
        user_id = self.user_ids[index]
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

        random.shuffle(clicked_texts)
        random.shuffle(browsed_texts)

        candidate_text, clicked_texts = clicked_texts[0], clicked_texts[1 : self.max_articles]
        browsed_texts = browsed_texts[:4]

        clicked_texts = clicked_texts + [""] * (self.max_articles - len(clicked_texts))
        browsed_texts = (browsed_texts * (4 // len(browsed_texts)))[:4]

        candidate_tokens = self.tokenizer(
            [candidate_text],
            return_tensors="pt",
            truncation=True,
            max_length=self.sequence_size,
            padding="max_length",
        )

        clicked_tokens = self.tokenizer(
            clicked_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.sequence_size,
            padding="max_length",
        )

        browsed_tokens = self.tokenizer(
            browsed_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.sequence_size,
            padding="max_length",
        )

        return candidate_tokens, clicked_tokens, browsed_tokens


if __name__ == "__main__":
    dataframe = pd.read_csv(directories.train_dataset_csv)
    dataset = OheadlineDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=batch_encoding_collate)
    doc_encoder = models.v1.DocEncoder()
    for x1, x2, x3 in dataloader:
        input_ids, attention_mask = x2["input_ids"], x2["attention_mask"]
        # input_ids: Tensor: (64, 32, 20)
        # attention_mask: Tensor: (64, 32, 20)

        i_batch, i_count, i_seq_len = input_ids.shape
        i_input_ids = input_ids.view(i_batch * i_count, i_seq_len)
        # i_input_ids: Tensor: (2048, 20)
        # i_attention_mask: Tensor: (2048, 20)

        m_batch, m_count, m_seq_len = attention_mask.shape
        m_attention_mask = attention_mask.view(m_batch * m_count, m_seq_len)
        print(i_input_ids.shape, m_attention_mask.shape)
        # torch.Size([2048, 20]) torch.Size([2048, 20])

        embeddings = doc_encoder(i_input_ids, m_attention_mask)
        # torch.Size([2048, 20, 768])
        # torch.Size([2048, 20])
        print(embeddings.shape)
