import numpy as np
import pytest
import pandas as pd
from torch.utils.data import DataLoader

import directories
from datasets.v1 import OheadlineDataset


def generate_sample_dataframe():
    """데모 데이터 프레임 생성 함수"""
    np.random.seed(42)  # 원하는 시드 값으로 변경 가능
    dataframe = pd.read_csv(directories.test_dataset_csv)
    return dataframe


@pytest.fixture
def sample_dataset():
    sample_df = generate_sample_dataframe()  # 100 개의 샘플 데이터 생성
    return OheadlineDataset(dataframe=sample_df)


def test_dataset_length(sample_dataset):
    assert len(sample_dataset) == 128  # 128 개의 고유한 사용자


def test_get_item_structure(sample_dataset):
    data = sample_dataset[0]
    assert isinstance(data, tuple)
    assert len(data) == 3  # candidate_tokens, clicked_tokens, browsed_tokens

    # 데이터 구조 확인
    for tokens in data:
        assert "input_ids" in tokens
        assert "attention_mask" in tokens


def test_dataloader_iteration():
    sample_df = generate_sample_dataframe()
    dataset = OheadlineDataset(dataframe=sample_df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        assert len(batch) == 3  # candidate_tokens, clicked_tokens, browsed_tokens

        for tokens in batch:
            assert "input_ids" in tokens
            assert "attention_mask" in tokens


def test_dataloader_batch_shape(sample_dataset):
    sample_df = generate_sample_dataframe()
    dataset = OheadlineDataset(dataframe=sample_df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Expected shapes
    # (users, articles, words)
    expected_candidate = {"input_ids": (64, 1, 20), "attention_mask": (64, 1, 20)}
    expected_clicked = {"input_ids": (64, 32, 20), "attention_mask": (64, 32, 20)}
    expected_browsed = {"input_ids": (64, 4, 20), "attention_mask": (64, 4, 20)}

    # Verify shapes for each token set
    candidate, clicked, browsed = next(dataloader.__iter__())
    assert tuple(candidate["input_ids"].shape) == expected_candidate["input_ids"]
    assert tuple(candidate["attention_mask"].shape) == expected_candidate["attention_mask"]

    assert tuple(clicked["input_ids"].shape) == expected_clicked["input_ids"]
    assert tuple(clicked["attention_mask"].shape) == expected_clicked["attention_mask"]

    assert tuple(browsed["input_ids"].shape) == expected_browsed["input_ids"]
    assert tuple(browsed["attention_mask"].shape) == expected_browsed["attention_mask"]
