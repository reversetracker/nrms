import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)


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
    def __init__(self, emb_dim: int, n_head: int, dropout: float = 0.2):
        super(NewsEncoder, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            emb_dim, n_head, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.additional_attn = nn.Parameter(torch.randn(emb_dim))
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_masks: torch.Tensor = None,
        softmax_masks: torch.Tensor = None,
    ):
        logger.debug(f"x shape: {x.shape}")
        # x shape: torch.Size([users * articles, article title word size, word embedding size])
        # x shape: torch.Size([8192, 20, 128])

        # key_padding_masks.shape: ([users * articles, seq_length])
        assert key_padding_masks.shape == (x.shape[0], x.shape[1])
        # softmax_masks.shape: ([users * articles, 1])
        assert softmax_masks.shape == (x.shape[0], 1)

        attn_output, _ = self.multi_head_attention(x, x, x, key_padding_mask=key_padding_masks)
        attn_output = self.norm_1(attn_output)
        logger.debug(f"attn_output shape: {attn_output.shape}")
        # attn_output shape: torch.Size([8192, 20, 128])

        # attention output 을 새로운 차원으로 projection 한다.
        # 데이터의 특징을 재조정 하고 필요한 정보를 과장 및 필요 없는 정보를 축소
        fc_output = self.fc(attn_output)
        fc_output = self.norm_2(fc_output)
        logger.debug(f"fc_output shape: {fc_output.shape}")
        # fc_output shape: torch.Size([8192, 20, 128])

        # tanh 를 통해 데이터의 특징에 비선형성 추가.
        tanh_output = torch.tanh(fc_output)
        logger.debug(f"tanh_output shape: {tanh_output.shape}")
        # tanh_output shape: torch.Size([8192, 20, 128])

        # Additional Attention
        # word embedding 에 weight 을 곱해서 스칼라로 만듬 즉 각 단어의 중요도를 나타냄
        # 20 단어, 8192 (배치), 128 (임베딩 사이즈) -> 128 임베딩이 스칼라(중요도)로 변환 되며
        # 즉 20개의 단어들의 중요도를 scalar 로 표현
        additional_attn_output = tanh_output.matmul(self.additional_attn)
        additional_attn_output = self.dropout(additional_attn_output)
        logger.debug(f"additional_attn_output shape: {additional_attn_output.shape}")
        # additional_attn_output shape: torch.Size([8192, 20])

        # 각 단어의 중요도를 softmax 를 통해 확률로 만듬
        softmax_output = F.softmax(additional_attn_output, dim=1)
        if softmax_masks is not None:
            softmax_output = softmax_output * softmax_masks

        logger.debug(f"softmax_output shape: {softmax_output.shape}")
        # softmax_output shape: torch.Size([8192, 20])

        # shape 체크.
        logger.debug("softmax_output.unsqueeze(-1) shape:", softmax_output.unsqueeze(-1).shape)
        # softmax_output.unsqueeze(-1) shape: torch.Size([8192, 20, 1])
        logger.debug("attn_output shape:", attn_output.shape)
        # attn_output shape: torch.Size([8192, 20, 128])

        # Aggregate the Output
        # 각단어의 중요도를 attention_output 에 곱해서 각 단어의 중요도에 따라서
        # attention_output 을 조정
        out = torch.sum(softmax_output.unsqueeze(-1) * attn_output, dim=1)
        logger.debug(f"out shape: {out.shape}")
        # out shape: torch.Size([8192, 128])
        return out


class UserEncoder(nn.Module):
    def __init__(self, emb_dim: int, n_head: int, dropout: float = 0.2):
        super(UserEncoder, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            emb_dim, n_head, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.additional_attn = nn.Parameter(torch.randn(emb_dim))
        self.norm_1 = nn.LayerNorm(emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        logger.debug(f"x shape: {x.shape}")
        # x shape: torch.Size([128, 64, 128])

        # multi head attention 을 이용하여
        # 'news_encoder 출력벡터' 의 attention 벡터로 변환
        attn_output, _ = self.multi_head_attention(x, x, x)
        attn_output = self.norm_1(attn_output)
        logger.debug(f"attn_output shape: {attn_output.shape}")
        # attn_output shape: torch.Size([128, 64, 128])

        # attention vector를 projection 하여 데이터의 특징을 재조정
        fc_output = self.fc(attn_output)
        fc_output = self.norm_2(fc_output)
        logger.debug(f"fc_output shape: {fc_output.shape}")
        # fc_output shape: torch.Size([128, 64, 128])

        # tanh 함수를 사용하여 데이터에 비선형성을 추가합니다.
        tanh_output = torch.tanh(fc_output)
        logger.debug(f"tanh_output shape: {tanh_output.shape}")
        # tanh_output shape: torch.Size([128, 64, 128])

        # 각 기사 벡터를 중요도 scalar로 변환 하기 위해 additional_attn 과 닷 프로덕트 연산.
        additional_attn_output = tanh_output.matmul(self.additional_attn)
        additional_attn_output = self.dropout(additional_attn_output)
        logger.debug(f"additional_attn_output shape: {additional_attn_output.shape}")
        # query_output shape: torch.Size([128, 64])

        # 각 기사벡터 중요도를 softmax 를 통해 확률로 변환합니다.
        attn_weight = F.softmax(additional_attn_output, dim=1)
        logger.debug(f"attention_weights shape: {attn_weight.shape}")
        # attention_weights shape: torch.Size([128, 64])

        # additional attention 을 attn_output 에 곱하여 attention 에 추가 가중치를 부여합니다.
        weighted_attention = attn_weight.unsqueeze(-1) * attn_output
        logger.debug(f"weighted_attention shape: {weighted_attention.shape}")
        # torch.Size([128, 64, 128]) users, articles, embed_dim
        # 128명의 유저들이 각각 64개의 기사를 보았고 각각의 기사 벡터는 128 dim 을 가지고 있음

        # user가 본 기사들의 벡터를 모두 더하여 user 벡터를 생성합니다.
        # (users, articles, embed_dim) -> (users, embed_dim)
        out = torch.sum(weighted_attention, dim=1)
        logger.debug(f"out shape: {out.shape}")
        # torch.Size([128, 128])
        return out


class NRMS(pl.LightningModule):
    def __init__(self, embed_size, num_heads):
        super(NRMS, self).__init__()
        self.news_encoder = NewsEncoder(embed_size, num_heads)
        self.user_encoder = UserEncoder(embed_size, num_heads)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, titles, key_padding_masks, softmax_masks):
        users, articles, seq_length, embed_size = titles.shape

        reshaped_titles = titles.view(users * articles, seq_length, embed_size)
        reshaped_key_padding_masks = key_padding_masks.view(users * articles, seq_length)
        reshaped_softmax_masks = softmax_masks.view(users * articles, 1)

        news_output = self.news_encoder(
            reshaped_titles, reshaped_key_padding_masks, reshaped_softmax_masks
        )
        news_output = news_output.view(users, articles, embed_size)
        user_output = self.user_encoder(news_output)

        scores = torch.bmm(news_output, user_output.unsqueeze(2)).squeeze(2)
        return scores

    def training_step(self, batch, batch_idx):
        titles, labels, key_padding_masks, softmax_masks = batch
        scores = self.forward(titles, key_padding_masks, softmax_masks)
        loss = self.criterion(scores, labels.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("avg_val_loss", avg_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "avg_val_loss",
            },
        }
