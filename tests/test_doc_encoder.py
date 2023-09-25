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


def test_forward_doc_encoder_with_clicked_tokens(sample_dataloader):
    clicked_tokens, _, __ = next(sample_dataloader.__iter__())

    clicked_input_ids = clicked_tokens["input_ids"]
    clicked_attention_mask = clicked_tokens["attention_mask"]

    nrms = models.v1.NRMS()
    embeddings = nrms.forward_doc_encoder(clicked_input_ids, clicked_attention_mask)

    assert embeddings.shape == (64, 32, 20, 768), "Invalid shape for embeddings"


def test_forward_doc_encoder_with_labeled_tokens(sample_dataloader):
    _, labeled_tokens, __ = next(sample_dataloader.__iter__())

    labeled_input_ids = labeled_tokens["input_ids"]
    labeled_attention_mask = labeled_tokens["attention_mask"]

    nrms = models.v1.NRMS()
    embeddings = nrms.forward_doc_encoder(labeled_input_ids, labeled_attention_mask)

    assert embeddings.shape == (64, 5, 20, 768), "Invalid shape for embeddings"
