import logging

import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from torch import optim
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

        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, head_dim)

        scores = torch.einsum("ijkl,ijml->ijkm", q, k) / (head_dim**0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.einsum("ijkm,ijml->ijkl", attn, v)

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

    def forward(
        self,
        x: torch.Tensor,
        key_padding_masks: torch.Tensor = None,
        softmax_masks: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logger.debug(f"x shape: {x.shape}")

        assert key_padding_masks.shape == (x.shape[0], x.shape[1])
        assert softmax_masks.shape == (x.shape[0], 1)

        attn_output, attn_output_weights = self.multi_head_attention(
            x, x, x, key_padding_mask=key_padding_masks
        )
        attn_output = self.norm_1(attn_output)
        logger.debug(f"attn_output shape: {attn_output.shape}")

        fc_output = self.fc(attn_output)
        fc_output = self.norm_2(fc_output)
        logger.debug(f"fc_output shape: {fc_output.shape}")

        tanh_output = torch.tanh(fc_output)
        logger.debug(f"tanh_output shape: {tanh_output.shape}")

        additional_attn_output = tanh_output.matmul(self.additional_attn)
        additional_attn_output = self.dropout(additional_attn_output)
        logger.debug(f"additional_attn_output shape: {additional_attn_output.shape}")

        softmax_output = F.softmax(additional_attn_output, dim=1)
        if softmax_masks is not None:
            softmax_output = softmax_output * softmax_masks
        logger.debug(f"softmax_output shape: {softmax_output.shape}")

        out = torch.sum(softmax_output.unsqueeze(-1) * attn_output, dim=1)
        logger.debug(f"out shape: {out.shape}")
        return out, attn_output_weights, softmax_output


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

        attn_output, _ = self.multi_head_attention(x, x, x)
        attn_output = self.norm_1(attn_output)
        logger.debug(f"attn_output shape: {attn_output.shape}")

        fc_output = self.fc(attn_output)
        fc_output = self.norm_2(fc_output)
        logger.debug(f"fc_output shape: {fc_output.shape}")

        tanh_output = torch.tanh(fc_output)
        logger.debug(f"tanh_output shape: {tanh_output.shape}")

        additional_attn_output = tanh_output.matmul(self.additional_attn)
        additional_attn_output = self.dropout(additional_attn_output)
        logger.debug(f"additional_attn_output shape: {additional_attn_output.shape}")

        attn_weight = F.softmax(additional_attn_output, dim=1)
        logger.debug(f"attention_weights shape: {attn_weight.shape}")

        weighted_attention = attn_weight.unsqueeze(-1) * attn_output
        logger.debug(f"weighted_attention shape: {weighted_attention.shape}")

        out = torch.sum(weighted_attention, dim=1)
        logger.debug(f"out shape: {out.shape}")
        return out


class NRMS(pl.LightningModule):
    def __init__(self, embed_size, num_heads):
        super(NRMS, self).__init__()

        self.news_encoder = NewsEncoder(embed_size, num_heads)
        self.user_encoder = UserEncoder(embed_size, num_heads)
        self.criterion = nn.BCEWithLogitsLoss()

        self.training_step_outputs = []
        self.validating_step_outputs = []
        self.testing_step_outputs = []

        self.save_hyperparameters()

    def forward(self, titles, key_padding_masks, softmax_masks):
        users, articles, seq_length, embed_size = titles.shape

        reshaped_titles = titles.view(users * articles, seq_length, embed_size)
        reshaped_key_padding_masks = key_padding_masks.view(users * articles, seq_length)
        reshaped_softmax_masks = softmax_masks.view(users * articles, 1)

        news_output, attn_weights, additive_attn_weights = self.news_encoder(
            reshaped_titles, reshaped_key_padding_masks, reshaped_softmax_masks
        )
        news_output = news_output.view(users, articles, embed_size)
        user_output = self.user_encoder(news_output)

        scores = torch.bmm(news_output, user_output.unsqueeze(2)).squeeze(2)
        return scores, attn_weights, additive_attn_weights

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "avg_val_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        titles, labels, key_padding_masks, softmax_masks = batch
        scores, _, __ = self.forward(titles, key_padding_masks, softmax_masks)
        loss = self.criterion(scores, labels.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        titles, labels, key_padding_masks, softmax_masks = batch
        scores, attn_weights, additive_softmax = self.forward(titles, key_padding_masks, softmax_masks)
        loss = self.criterion(scores, labels.float())
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.validating_step_outputs.append(loss)

        # attention visualization logging here..
        fig, ax = plt.subplots(figsize=(10, 10))
        specific_attn_weights = attn_weights[0]
        sns.heatmap(specific_attn_weights.cpu().detach().numpy(), ax=ax, cmap="viridis")
        ax.set_title("Attention Weights")
        wandb.log(
            {
                "attention_weights": [
                    wandb.Image(fig, caption=f"Attention Weights Batch-{batch_idx}")
                ]
            }
        )
        plt.close(fig)

        # additive softmax_results visualization logging
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(additive_softmax.cpu().detach().numpy(), ax=ax, cmap="viridis")
        ax.set_title("Additive Weights")
        wandb.log(
            {
                "additive_softmax": [
                    wandb.Image(fig, caption=f"Additive Softmax Batch-{batch_idx}")
                ]
            }
        )
        plt.close(fig)

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        titles, labels, key_padding_masks, softmax_masks = batch
        scores, _, __ = self.forward(titles, key_padding_masks, softmax_masks)
        loss = self.criterion(scores, labels.float())

        self.log("test_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.testing_step_outputs.append(loss)
        return {"test_loss": loss}

    def on_train_epoch_end(self) -> None:
        avg_loss = torch.stack([x for x in self.training_step_outputs]).mean()
        self.log("avg_train_loss", avg_loss, prog_bar=True)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack([x for x in self.validating_step_outputs]).mean()  # 수정된 부분
        self.log("avg_val_loss", avg_loss, prog_bar=True)
        self.validating_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        avg_loss = torch.stack([x for x in self.testing_step_outputs]).mean()
        self.log("avg_test_loss", avg_loss, prog_bar=True)
        self.testing_step_outputs.clear()
