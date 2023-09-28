import random

import pandas as pd
import torch
from google.oauth2.service_account import Credentials
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer

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

    def only_active_users(group):
        has_viewed_true_count = group[group["has_viewed"] == True].shape[0]
        has_viewed_false_count = group[group["has_viewed"] == False].shape[0]
        return has_viewed_true_count > 5 and has_viewed_false_count > 10

    df = df.groupby("user_id").filter(only_active_users)
    df.to_csv(directories.train_dataset_csv, index=False)


class OheadlineDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        article_size: int = 32,
        sequence_size: int = 20,
        embed_dim: int = 768,
        K: int = 4,
        tokenizer=None,
    ):
        self.dataframe = dataframe
        self.user_ids = self.dataframe["user_id"].unique()
        self.user_data_groups = dict(tuple(self.dataframe.groupby("user_id")))
        self.article_size = article_size
        self.sequence_size = sequence_size
        self.embed_dim = embed_dim
        self.K = K
        self.tokenizer = tokenizer or ElectraTokenizer.from_pretrained(
            "monologg/koelectra-base-v3-discriminator"
        )

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, index: int) -> tuple:
        user_id = self.user_ids[index]
        user_data = self.user_data_groups[user_id]

        titles = user_data["title"].values.tolist()
        has_viewed_list = user_data["has_viewed"].values.tolist()

        assert len(titles) == len(has_viewed_list)

        clicked_texts = [x[0] for x in zip(titles, has_viewed_list) if x[1] is True]
        browsed_texts = [
            x[0]
            for x in zip(titles, has_viewed_list)
            if x[1] is False and x[0] not in clicked_texts
        ]

        random.shuffle(clicked_texts)
        random.shuffle(browsed_texts)

        candidate_text = clicked_texts.pop(0)
        clicked_texts = clicked_texts[: self.article_size]
        browsed_texts = browsed_texts[: self.K]

        # ADD PADDINGS
        # max_articles[32] 개를 채우지 못한 경우 '빈 문자열'을 패딩으로 채워서 32개의 기사로 만듬
        clicked_texts = clicked_texts + [""] * (self.article_size - len(clicked_texts))
        # K[4] 개를 채우지 못한 경우 '중복' 기사로 채움
        browsed_texts = (browsed_texts * (self.K // len(browsed_texts)))[: self.K]

        # 1*P + K*N
        label_texts = [candidate_text] + browsed_texts
        random_index = random.randint(1, self.K - 1)
        label_texts[0], label_texts[random_index] = label_texts[random_index], label_texts[0]

        # CLICKED DATA 에 key_padding_mask, softmax_padding_mask 추가
        clicked_tokens = self.tokenizer(
            clicked_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.sequence_size,
            padding="max_length",
        )
        key_padding_mask = ~clicked_tokens["attention_mask"].bool()
        rows, features = key_padding_mask.shape
        false_counts = (key_padding_mask == False).sum(dim=1).reshape(rows, 1)
        softmax_padding_mask = torch.where(
            false_counts <= 2,  # 2, 3, 0, 0, 0
            torch.zeros_like(torch.Tensor(features), dtype=torch.int),
            torch.ones_like(torch.Tensor(features), dtype=torch.int),
        )
        clicked_tokens["key_padding_mask"] = key_padding_mask
        clicked_tokens["softmax_padding_mask"] = softmax_padding_mask

        # LABELED DATA 에 key_padding_mask 추가
        labeled_tokens = self.tokenizer(
            label_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.sequence_size,
            padding="max_length",
        )
        key_padding_mask = ~labeled_tokens["attention_mask"].bool()
        labeled_tokens["key_padding_mask"] = key_padding_mask

        cross_entropy_labels = torch.Tensor([random_index])

        return clicked_tokens, labeled_tokens, cross_entropy_labels


if __name__ == "__main__":
    dataframe = pd.read_csv(directories.unittest_dataset_csv)
    dataset = OheadlineDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for batch in dataloader:
        clicked_tokens, labeled_tokens, labels = batch
        print(clicked_tokens)
        print(labeled_tokens)
        print(labels)
        break
