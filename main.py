import torch

import models

users = 128
articles = 64
seq_length = 20
embed_size = 768
num_heads = 8

dataset = torch.rand((users, articles, seq_length, embed_size))
labels = torch.randint(0, 2, (users, articles))
print(f"dataset shape: {dataset.shape}")
# dataset shape: torch.Size([128, 64, 20, 768])
print(f"labels shape: {labels.shape}")
# labels shape: torch.Size([128, 64])

dataset_reshaped = dataset.view(users * articles, seq_length, embed_size)
labels_reshaped = labels.view(users * articles)
print(f"dataset_reshaped shape: {dataset_reshaped.shape}")
# dataset_reshaped shape: torch.Size([8192, 20, 768])
print(f"labels_reshaped shape: {labels_reshaped.shape}")
# labels_reshaped shape: torch.Size([8192])

news_encoder = models.v1.NewsEncoder(embed_size, num_heads)
user_encoder = models.v1.UserEncoder(embed_size, num_heads)

news_output = news_encoder(dataset_reshaped)
print(f"news_output shape: {news_output.shape}")
# news_output shape: torch.Size([8192, 768])

reshaped_news_output = news_output.view(users, articles, embed_size)
print(f"reshaped_news_output shape: {reshaped_news_output.shape}")
# reshaped_news_output shape: torch.Size([128, 64, 768])

user_output = user_encoder(reshaped_news_output)
print(f"user_output shape: {user_output.shape}")
# user_output shape: torch.Size([128, 768])

scores = torch.bmm(reshaped_news_output, user_output.unsqueeze(2))
scores = scores.squeeze(2)

print(f"scores shape: {scores.shape}")
# scores shape: torch.Size([128, 64])
