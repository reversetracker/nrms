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


def test_forward_user_encoder(sample_dataloader):
    nrms = models.v1.NRMS()

    candidate, clicked, browsed = next(sample_dataloader.__iter__())

    # CANDIDATE
    candidate_input_ids = candidate["input_ids"]
    candidate_attention_mask = candidate["attention_mask"]

    candidate_embeddings = nrms.forward_user_encoder(candidate_input_ids, candidate_attention_mask)
    # (users, encoder_dim)
    assert candidate_embeddings.shape == (64, 128)

    # CLICKED
    clicked_input_ids = clicked["input_ids"]
    clicked_attention_mask = clicked["attention_mask"]

    clicked_embeddings = nrms.forward_user_encoder(clicked_input_ids, clicked_attention_mask)
    # (users, encoder_dim)
    assert clicked_embeddings.shape == (64, 128)

    # BROWSED
    browsed_input_ids = browsed["input_ids"]
    browsed_attention_mask = browsed["attention_mask"]

    browsed_embeddings = nrms.forward_user_encoder(browsed_input_ids, browsed_attention_mask)
    # (users, encoder_dim)
    assert browsed_embeddings.shape == (64, 128)
