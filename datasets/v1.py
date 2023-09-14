import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import directories
import tokenizer


class OheadlineDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_articles: int = 64,
        sequence_size: int = 20,
        embedding_dim: int = 768,
    ):
        self.dataframe = dataframe
        self.user_ids = self.dataframe["user_id"].unique()
        self.max_articles = max_articles
        self.sequence_size = sequence_size
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        user_data = self.dataframe[self.dataframe["user_id"] == user_id]
        user_data = user_data.dropna(subset=["article_title", "has_viewed"])

        title_list = user_data["article_title"].values.tolist()[: self.max_articles]
        has_viewed_list = user_data["has_viewed"].values.tolist()[: self.max_articles]

        title_embeddings, masks = tokenizer.texts_to_embeddings(title_list)

        # zero padding 으로 batch size 결정.
        titles_tensor = torch.zeros(
            (self.max_articles, self.sequence_size, self.embedding_dim), dtype=torch.float
        )
        has_viewed_tensor = torch.zeros(self.max_articles, dtype=torch.long)
        masks_tensor = torch.zeros((self.max_articles, self.sequence_size), dtype=torch.bool)

        # zero padding 에 override
        titles_tensor[: len(title_list)] = title_embeddings
        has_viewed_tensor[: len(has_viewed_list)] = torch.tensor(has_viewed_list)
        masks_tensor[: len(title_list)] = masks
        masks_tensor = ~masks_tensor

        return titles_tensor, has_viewed_tensor, masks_tensor


class OheadlineDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_articles: int = 64,
        sequence_size: int = 20,
        embedding_dim: int = 768,
    ):
        self.dataframe = dataframe.dropna(subset=["article_title", "has_viewed"]).sort_values(
            by="user_id"
        )
        self.user_ids = self.dataframe["user_id"].unique()
        self.user_data_groups = dict(tuple(self.dataframe.groupby("user_id")))
        self.max_articles = max_articles
        self.sequence_size = sequence_size
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        user_data = self.user_data_groups[user_id]

        title_list = user_data["article_title"].values.tolist()[: self.max_articles]
        has_viewed_list = user_data["has_viewed"].values.tolist()[: self.max_articles]

        title_embeddings, masks = tokenizer.texts_to_embeddings(title_list)

        titles_tensor = torch.zeros(
            (self.max_articles, self.sequence_size, self.embedding_dim), dtype=torch.float
        )
        has_viewed_tensor = torch.zeros(self.max_articles, dtype=torch.long)
        masks_tensor = torch.ones((self.max_articles, self.sequence_size), dtype=torch.bool)

        titles_tensor[: len(title_list)] = title_embeddings
        has_viewed_tensor[: len(has_viewed_list)] = torch.tensor(has_viewed_list)
        masks_tensor[: len(title_list)] = masks

        # torch.Size([64, 20, 768])
        # torch.Size([64])
        # torch.Size([64, 20])
        return titles_tensor, has_viewed_tensor, masks_tensor


class OheadlineDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_articles: int = 64,
        sequence_size: int = 20,
        embedding_dim: int = 768,
    ):
        self.dataframe = dataframe.dropna(subset=["article_title", "has_viewed"]).sort_values(
            by="user_id"
        )
        self.dataframe = tokenizer.precompute_embeddings(
            self.dataframe, "article_title", "embedding"
        )

        self.user_ids = self.dataframe["user_id"].unique()
        self.user_data_groups = dict(tuple(self.dataframe.groupby("user_id")))
        self.max_articles = max_articles
        self.sequence_size = sequence_size
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        user_data = self.user_data_groups[user_id]

        title_embeddings_list = user_data["embedding"].values.tolist()[: self.max_articles]
        title_embeddings_list = np.array(title_embeddings_list)

        has_viewed_list = user_data["has_viewed"].values.tolist()[: self.max_articles]
        has_viewed_list = np.array(has_viewed_list)

        mask_list = user_data["mask"].values.tolist()[: self.max_articles]
        mask_list = np.array(mask_list)

        titles_tensor = torch.zeros(
            (self.max_articles, self.sequence_size, self.embedding_dim), dtype=torch.float
        )
        has_viewed_tensor = torch.zeros(self.max_articles, dtype=torch.long)
        masks_tensor = torch.ones((self.max_articles, self.sequence_size), dtype=torch.bool)

        titles_tensor[: len(title_embeddings_list)] = torch.tensor(
            title_embeddings_list, dtype=torch.float
        )
        has_viewed_tensor[: len(has_viewed_list)] = torch.tensor(has_viewed_list)
        masks_tensor[: len(mask_list)] = torch.tensor(mask_list, dtype=torch.bool)

        return titles_tensor, has_viewed_tensor, masks_tensor


if __name__ == "__main__":
    dataframe = pd.read_csv(directories.bq_results)
    dataset = OheadlineDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for titles, labels, masks in dataloader:
        print(titles.shape)
        # torch.Size([64, 64, 20, 128])
        print(labels.shape)
        # torch.Size([64, 64])
        print(masks.shape)
        # torch.Size([64, 64, 20])
