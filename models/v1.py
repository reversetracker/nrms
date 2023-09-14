import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import pytorch_lightning as pl


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.num_heads = num_heads

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        head_dim = d_model // self.num_heads

        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, head_dim)

        # Scaled dot-product attention
        scores = torch.einsum("ijkl,ijml->ijkm", q, k) / (head_dim**0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.einsum("ijkm,ijml->ijkl", attn, v)

        # Concatenation of heads
        context = context.contiguous().view(batch_size, seq_len, d_model)
        return context


class NewsEncoder(nn.Module):
    def __init__(self, emb_dim: int, n_head: int, dropout: float = 0.1):
        super(NewsEncoder, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            emb_dim, n_head, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.additional_attn = nn.Parameter(torch.randn(emb_dim))
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        print(f"x shape: {x.shape}")
        # x shape: torch.Size([users * articles, article title word size, word embedding size])
        # x shape: torch.Size([8192, 20, 128])

        all_true_rows = key_padding_mask.all(dim=1)
        key_padding_mask[all_true_rows] = False
        attn_output, _ = self.multi_head_attention(x, x, x, key_padding_mask=key_padding_mask)

        # with residual net
        attn_output = self.norm_1(attn_output + x)
        print(f"attn_output shape: {attn_output.shape}")
        # attn_output shape: torch.Size([8192, 20, 128])

        # without residual net
        # attn_output = self.norm1(attn_output)
        # print(f"attn_output shape: {attn_output.shape}")

        # attention output 을 새로운 차원으로 projection 한다.
        # 데이터의 특징을 재조정 하고 필요한 정보를 과장 및 필요 없는 정보를 축소
        fc_output = self.fc(attn_output)
        fc_output = self.norm_2(fc_output)
        print(f"fc_output shape: {fc_output.shape}")
        # fc_output shape: torch.Size([8192, 20, 128])

        # tanh 를 통해 데이터의 특징에 비선형성 추가.
        tanh_output = torch.tanh(fc_output)
        print(f"tanh_output shape: {tanh_output.shape}")
        # tanh_output shape: torch.Size([8192, 20, 128])

        # Additional Attention
        # word embedding 에 weight 을 곱해서 스칼라로 만듬 즉 각 단어의 중요도를 나타냄
        # 20 단어, 8192 (배치), 128 (임베딩 사이즈) -> 128 임베딩이 스칼라(중요도)로 변환 되며
        # 즉 20개의 단어들의 중요도를 scalar 로 표현
        additional_attn_output = tanh_output.matmul(self.additional_attn)
        additional_attn_output = self.dropout(additional_attn_output)
        print(f"additional_attn_output shape: {additional_attn_output.shape}")
        # additional_attn_output shape: torch.Size([8192, 20])

        # 각 단어의 중요도를 softmax 를 통해 확률로 만듬
        mask_for_softmax = (~all_true_rows).float().unsqueeze(-1)
        softmax_output = F.softmax(additional_attn_output, dim=1)
        softmax_output = softmax_output * mask_for_softmax

        print(f"softmax_output shape: {softmax_output.shape}")
        # softmax_output shape: torch.Size([8192, 20])

        # shape 체크.
        print("softmax_output.unsqueeze(-1) shape:", softmax_output.unsqueeze(-1).shape)
        # softmax_output.unsqueeze(-1) shape: torch.Size([8192, 20, 1])
        print("attn_output shape:", attn_output.shape)
        # attn_output shape: torch.Size([8192, 20, 128])

        # Aggregate the Output
        # 각단어의 중요도를 attention_output 에 곱해서 각 단어의 중요도에 따라서
        # attention_output 을 조정
        out = torch.sum(softmax_output.unsqueeze(-1) * attn_output, dim=1)
        print(f"out shape: {out.shape}")
        # out shape: torch.Size([8192, 128])
        return out


class UserEncoder(nn.Module):
    def __init__(self, emb_dim: int, n_head: int, dropout: float = 0.1):
        super(UserEncoder, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            emb_dim, n_head, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.additional_attn = nn.Parameter(torch.randn(emb_dim))

    def forward(self, x):
        print(f"x shape: {x.shape}")
        # x shape: torch.Size([128, 64, 128])

        # multi head attention 을 이용하여
        # news_encoder output embedding 의 attention 벡터로 변환
        attn_output, _ = self.multi_head_attention(x, x, x)
        print(f"attn_output shape: {attn_output.shape}")
        # attn_output shape: torch.Size([128, 64, 128])

        # attention vector를 projection 하여 데이터의 특징을 재조정
        fc_output = self.fc(attn_output)
        print(f"fc_output shape: {fc_output.shape}")
        # fc_output shape: torch.Size([128, 64, 128])

        # tanh 함수를 사용하여 데이터에 비선형성을 추가합니다.
        tanh_output = torch.tanh(fc_output)
        print(f"tanh_output shape: {tanh_output.shape}")
        # tanh_output shape: torch.Size([128, 64, 128])

        # 각 기사 벡터를 중요도 scalar로 변환 하기 위해 additional_attn 과 닷 프로덕트 연산.
        additional_attn_output = tanh_output.matmul(self.additional_attn)
        print(f"additional_attn_output shape: {additional_attn_output.shape}")
        # query_output shape: torch.Size([128, 64])

        # 각 기사벡터 중요도를 softmax 를 통해 확률로 변환합니다.
        attn_weight = F.softmax(additional_attn_output, dim=1)
        print(f"attention_weights shape: {attn_weight.shape}")
        # attention_weights shape: torch.Size([128, 64])

        # additional attention 을 attn_output 에 곱하여 attention 에 추가 가중치를 부여합니다.
        weighted_attention = attn_weight.unsqueeze(-1) * attn_output
        print(f"weighted_attention shape: {weighted_attention.shape}")
        # torch.Size([128, 64, 128]) users, articles, embed_dim
        # 128명의 유저들이 각각 64개의 기사를 보았고 각각의 기사 벡터는 128 dim 을 가지고 있음

        # user가 본 기사들의 벡터를 모두 더하여 user 벡터를 생성합니다.
        # (users, articles, embed_dim) -> (users, embed_dim)
        out = torch.sum(weighted_attention, dim=1)
        print(f"out shape: {out.shape}")
        # torch.Size([128, 128])
        return out


class NRMS(pl.LightningModule):
    def __init__(self, embed_size, num_heads):
        super(NRMS, self).__init__()
        self.news_encoder = NewsEncoder(embed_size, num_heads)
        self.user_encoder = UserEncoder(embed_size, num_heads)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, titles, masks):
        users, articles, seq_length, embed_size = titles.shape

        reshaped_titles = titles.view(users * articles, seq_length, embed_size)
        reshaped_masks = masks.view(users * articles, seq_length)

        news_output = self.news_encoder(reshaped_titles, reshaped_masks)
        news_output = news_output.view(users, articles, embed_size)
        user_output = self.user_encoder(news_output)

        scores = torch.bmm(news_output, user_output.unsqueeze(2)).squeeze(2)
        return scores

    def training_step(self, batch, batch_idx):
        titles, labels, mask = batch
        scores = self.forward(titles, mask)
        loss = self.criterion(scores, labels.float())
        print(f"train_loss: {loss}")
        return loss

    def configure_optimizers(self):
        parameters = list(self.news_encoder.parameters()) + list(self.user_encoder.parameters())
        optimizer = optim.Adam(parameters, lr=0.0001, weight_decay=1e-5)
        return optimizer
