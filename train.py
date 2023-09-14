import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import datasets.v1
import directories
import models

torch.autograd.set_detect_anomaly(True)

users = 64  # batch_size
articles = 64  # article number per user read
seq_length = 20  # 기사당 단어 수
embed_size = 128  # 임베딩 차원
num_heads = 8  # 대가리 수

dataframe = pd.read_csv(directories.bq_results)
dataset = datasets.v1.OheadlineDataset(dataframe)

news_encoder = models.v1.NewsEncoder(embed_size, num_heads)
user_encoder = models.v1.UserEncoder(embed_size, num_heads)

news_parameters = list(news_encoder.parameters())
user_parameters = list(user_encoder.parameters())
parameters = news_parameters + user_parameters
optimizer = optim.Adam(parameters, lr=0.0001)

criterion = nn.BCEWithLogitsLoss()

epochs = 100
for epoch in range(epochs):
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for titles, labels, mask in dataloader:
        print(f"Batched titles shape: {titles.shape}")
        # Titles shape: torch.Size([64, 64, 20, 128])

        print(f"Batched labels shape: {labels.shape}")
        # Labels shape: torch.Size([64, 64])

        print(f"Batched mask shape: {mask.shape}")
        # Mask shape: torch.Size([64, 64, 20])

        reshaped_title = titles.view(users * articles, seq_length, embed_size)
        print(f"Reshaped titles shape: {reshaped_title.shape}")
        # Reshaped titles shape: torch.Size([4096, 20, 128])

        reshaped_mask = mask.view(users * articles, seq_length)
        print(f"Reshaped mask shape: {reshaped_mask.shape}")
        # Reshaped mask shape: torch.Size([4096, 20])

        news_output = news_encoder(reshaped_title, reshaped_mask)
        news_output = news_output.view(users, articles, embed_size)
        user_output = user_encoder(news_output)

        scores = torch.bmm(news_output, user_output.unsqueeze(2)).squeeze(2)
        print(f"Scores shape: {scores.shape}")
        # Scores shape: torch.Size([64, 64])

        loss = criterion(scores, labels.float())

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
