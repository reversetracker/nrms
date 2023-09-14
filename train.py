import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import datasets.v1
import directories
from models.v1 import NRMS

torch.autograd.set_detect_anomaly(True)

users = 64  # batch_size
articles = 64  # article number per user read
seq_length = 20  # 기사당 단어 수
embed_size = 768  # 임베딩 차원
num_heads = 8  # 대가리 수

dataframe = pd.read_csv(directories.bq_results)
dataset = datasets.v1.OheadlineDataset(dataframe)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = NRMS(embed_size, num_heads)

# Lightning trainer
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, dataloader)
