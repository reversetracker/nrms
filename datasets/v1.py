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

        assert len(title_list) == len(has_viewed_list)

        title_embeddings, mask_embeddings = tokenizer.texts_to_embeddings(title_list)
        has_viewed_embeddings = torch.tensor(has_viewed_list)

        titles_batch = torch.zeros(
            (self.max_articles, self.sequence_size, self.embedding_dim), dtype=torch.float
        )
        has_viewed_batch = torch.zeros(self.max_articles, dtype=torch.long)
        key_padding_masks_batch = torch.zeros(
            (self.max_articles, self.sequence_size), dtype=torch.bool
        )
        softmax_masks_batch = torch.ones((self.max_articles, 1))

        titles_batch[: len(title_list)] = title_embeddings
        has_viewed_batch[: len(has_viewed_list)] = has_viewed_embeddings
        key_padding_masks_batch[: len(title_list)] = mask_embeddings
        softmax_masks_batch[len(has_viewed_list) :] = 0

        # torch.Size([64, 20, 768])
        # torch.Size([64])
        # torch.Size([64, 20])
        # torch.Size([64, 1])
        return titles_batch, has_viewed_batch, key_padding_masks_batch, softmax_masks_batch


if __name__ == "__main__":
    dataframe = pd.read_csv(directories.bq_results_csv)
    dataset = OheadlineDataset(dataframe)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for titles, labels, key_padding_mask, softmax_mask in dataloader:
        print(titles.shape)
        # torch.Size([64, 64, 20, 128])
        print(labels.shape)
        # torch.Size([64, 64])
        print(key_padding_mask.shape)
        # torch.Size([64, 64, 20])
        print(softmax_mask.shape)
