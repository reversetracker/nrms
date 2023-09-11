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
        embedding_dim: int = 128,
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
        user_data = user_data.dropna(subset=["article_title", "watch_or_not"])

        titles = user_data["article_title"].values.tolist()
        watch_or_not_list = user_data["watch_or_not"].values.tolist()

        title_tensor_shape = (self.max_articles, self.sequence_size, self.embedding_dim)
        titles_tensor = torch.zeros(title_tensor_shape, dtype=torch.float)
        watch_or_not_tensor = torch.zeros(self.max_articles, dtype=torch.long)

        for idx, title in enumerate(titles[: self.max_articles]):
            titles_tensor[idx] = torch.FloatTensor(tokenizer.tokenize(title))

        for idx, watch in enumerate(watch_or_not_list[: self.max_articles]):
            watch_or_not_tensor[idx] = watch

        # 기사 수 마스크 계산
        sequence_mask = torch.ones(self.max_articles, dtype=torch.bool)
        sequence_mask[len(titles) :] = 0

        # 기사 제목 글자수 마스크 계산
        token_mask = torch.ones(self.max_articles, self.sequence_size, dtype=torch.bool)
        for idx, title in enumerate(titles[: self.max_articles]):
            num_tokens = len(title.split()) if title else 0
            token_mask[idx, num_tokens:] = 0

        combined_mask_tensor = sequence_mask.unsqueeze(-1) * token_mask
        combined_mask_tensor = ~combined_mask_tensor

        # titles_tensor: torch.Size([64, 20, 128])
        # watch_or_not_tensor: torch.Size([64])
        # combined_mask_tensor: torch.Size([64, 20])
        return titles_tensor, watch_or_not_tensor, combined_mask_tensor


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
