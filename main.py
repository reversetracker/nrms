import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import directories
import models.v1
from datasets.v1 import OheadlineDataset

BATCH_SIZE = 64  # users

SEQUENCE_LENGTH = 20  # max sequence length

EMBED_DIM: int = 768  # electra embeddings dimension

ENCODER_DIM: int = 128  # news encoder output dimension

NUM_HEADS: int = 8  # 머리 갯수

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
    clicked_tokens, labeled_tokens, _ = batch

    # CLICKED
    clicked_input_ids = clicked_tokens["input_ids"]
    clicked_attention_mask = clicked_tokens["attention_mask"]
    clicked_key_padding_mask = clicked_tokens["key_padding_mask"]
    clicked_softmax_padding_mask = clicked_tokens["softmax_padding_mask"]

    # LABELED
    labeled_input_ids = labeled_tokens["input_ids"]
    labeled_attention_mask = labeled_tokens["attention_mask"]
    labeled_key_padding_mask = labeled_tokens["key_padding_mask"]

    scores, c_weights, a_weights = nrms.forward(
        clicked_ids=clicked_input_ids,
        clicked_attention_mask=clicked_attention_mask,
        clicked_key_padding_mask=clicked_key_padding_mask,
        clicked_softmax_padding_mask=clicked_softmax_padding_mask,
        labeled_ids=labeled_input_ids,
        labeled_attention_mask=labeled_attention_mask,
        labeled_key_padding_mask=labeled_key_padding_mask,
    )

    assert scores.shape == (64, 5)
    assert c_weights.shape == (64 * 5, 20, 20)
    assert a_weights.shape == (64 * 5, 20)
