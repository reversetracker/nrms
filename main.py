import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

import datasets.v1
import directories
import models.v1
from datasets.v1 import OheadlineDataset

BATCH_SIZE = 64  # users

SEQUENCE_LENGTH = 20  # max sequence length

INPUT_DIM: int = 768  # electra embeddings dimension

OUTPUT_DIM: int = 128  # news encoder output dimension

NUM_HEADS: int = 8  # 귀두 갯수

np.random.seed(42)  # 원하는 시드 값으로 변경 가능

sample_df = pd.read_csv(directories.unittest_dataset_csv)
sample_dataset = OheadlineDataset(dataframe=sample_df)
sample_dataloader = DataLoader(
    sample_dataset,
    batch_size=64,
    shuffle=True,
)

nrms = models.v1.NRMS()

for batch in sample_dataloader:
    candidate, clicked, browsed = batch

    # CANDIDATE
    candidate_input_ids = candidate["input_ids"]
    candidate_attention_mask = candidate["attention_mask"]

    # CLICKED
    clicked_input_ids = clicked["input_ids"]
    clicked_attention_mask = clicked["attention_mask"]

    # BROWSED
    browsed_input_ids = browsed["input_ids"]
    browsed_attention_mask = browsed["attention_mask"]

    scores, c_weights, a_weights = nrms.forward(
        candidate_input_ids,
        candidate_attention_mask,
        clicked_input_ids,
        clicked_attention_mask,
        browsed_input_ids,
        browsed_attention_mask,
    )

    assert scores.shape == (64, 5)
    assert c_weights.shape == (64, 20, 20)
    assert a_weights.shape == (64, 20)
