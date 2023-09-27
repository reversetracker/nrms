import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

import directories
import models.v1
from datasets.v1 import OheadlineDataset

BATCH_SIZE = 64  # users

SEQUENCE_LENGTH = 20  # max sequence length

INPUT_DIM: int = 768  # electra embeddings dimension

OUTPUT_DIM: int = 128  # news encoder output dimension

NUM_HEADS: int = 8  # 귀두 갯수


@pytest.fixture
def sample_dataset():
    np.random.seed(42)  # 원하는 시드 값으로 변경 가능
    sample_df = pd.read_csv(directories.unittest_dataset_csv)
    return OheadlineDataset(dataframe=sample_df)


@pytest.fixture
def sample_dataloader(sample_dataset):
    return DataLoader(sample_dataset, batch_size=64, shuffle=True)


def test_forward_news_encoder(sample_dataloader):
    nrms = models.v1.NRMS()

    clicked_tokens, labeled_tokens, _ = next(sample_dataloader.__iter__())

    # CLICKED
    clicked_input_ids = clicked_tokens["input_ids"]
    clicked_attention_mask = clicked_tokens["attention_mask"]

    clicked_embeddings, _, __ = nrms.forward_news_encoder(
        clicked_input_ids, clicked_attention_mask
    )
    # (users, titles, encoder_dim)
    assert clicked_embeddings.shape == (64, 32, 128)

    # CANDIDATE
    candidate_input_ids = labeled_tokens["input_ids"]
    candidate_attention_mask = labeled_tokens["attention_mask"]

    candidate_embeddings, _, __ = nrms.forward_news_encoder(
        candidate_input_ids, candidate_attention_mask
    )
    # (users, titles, encoder_dim)
    assert candidate_embeddings.shape == (64, 5, 128)