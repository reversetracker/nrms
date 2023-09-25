import numpy as np
import pytest
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BatchEncoding

import directories
from datasets.v1 import OheadlineDataset


@pytest.fixture
def sample_dataset():
    np.random.seed(42)  # 원하는 시드 값으로 변경 가능
    sample_df = pd.read_csv(directories.unittest_dataset_csv)
    return OheadlineDataset(dataframe=sample_df)


def test_dataset_length(sample_dataset):
    assert len(sample_dataset) == 128  # 128 개의 고유한 사용자


def test_dataloader_iteration(sample_dataset):
    dataloader = DataLoader(sample_dataset, batch_size=64, shuffle=True)

    for batch in dataloader:
        clicked_tokens, labeled_tokens, labels = batch

        assert isinstance(clicked_tokens, BatchEncoding)
        assert clicked_tokens.input_ids.shape == torch.Size([64, 32, 20])
        assert clicked_tokens.attention_mask.shape == torch.Size([64, 32, 20])

        assert isinstance(labeled_tokens, BatchEncoding)
        assert labeled_tokens.input_ids.shape == torch.Size([64, 5, 20])
        assert labeled_tokens.attention_mask.shape == torch.Size([64, 5, 20])

        assert isinstance(labels, torch.Tensor)
        assert labels.shape == torch.Size([64, 1])
