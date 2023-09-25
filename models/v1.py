import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from transformers import ElectraModel, BatchEncoding

import utils

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


class DocEncoder(nn.Module):
    def __init__(
        self,
        max_length: int = 20,
        model_name: str = "monologg/koelectra-base-v3-discriminator",
    ):
        super(DocEncoder, self).__init__()
        self.model = ElectraModel.from_pretrained(model_name)
        self.max_length = max_length

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        return embeddings


class DeprecatedAdditiveAttention(nn.Module):
    """Additive Attention learns the importance of each word in the sequence."""

    def __init__(self, input_dim: int = 768, output_dim: int = 128):
        super(DeprecatedAdditiveAttention, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.query = nn.Parameter(torch.randn(output_dim))
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x: torch.Tensor):
        x_proj = self.linear(x)
        x_proj = self.tanh(x_proj)
        attn_scores = torch.matmul(x_proj, self.query)
        attn_probs = self.softmax(attn_scores)
        return attn_probs


class AdditiveAttention(nn.Module):
    """Additive Attention learns the importance of each word in the sequence."""

    def __init__(self, input_dim: int = 768, output_dim: int = 128):
        super(AdditiveAttention, self).__init__()

        self.proj = nn.Linear(input_dim, output_dim, bias=True)
        self.proj_v = nn.Linear(output_dim, 1, bias=False)  # No bias for this layer, similar to A.
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        nn.init.xavier_normal_(self.proj.weight)

    def forward(self, x: torch.Tensor):
        x_proj = self.proj(x)
        x_proj = self.tanh(x_proj)
        attn_scores = self.proj_v(x_proj).squeeze(-1)
        attn_probs = self.softmax(attn_scores)
        return attn_probs


class NewsEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super(NewsEncoder, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.additive_attention = AdditiveAttention(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

        nn.init.xavier_normal_(self.linear.weight)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_masks: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logger.debug(f"x shape: {x.shape}")

        assert key_padding_masks.shape == (x.shape[0], x.shape[1])

        # Multi-head attention
        context, context_weights = self.multi_head_attention(
            x, x, x, key_padding_mask=key_padding_masks
        )
        logger.debug(f"context shape: {context.shape}")

        # Fully connected layer
        transformed_context = self.linear(context)
        transformed_context = self.tanh(transformed_context)
        logger.debug(f"transformed context shape: {transformed_context.shape}")

        # Additive attention
        additive_weights = self.additive_attention(context)

        # Weighted context by the attention weights
        out = torch.sum(additive_weights.unsqueeze(-1) * transformed_context, dim=1)
        logger.debug(f"news encoder out shape: {out.shape}")
        return out, context_weights, additive_weights


class UserEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super(UserEncoder, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            input_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.additive_attention = AdditiveAttention(input_dim, input_dim)
        self.linear = nn.Linear(input_dim, input_dim)
        self.tanh = nn.Tanh()

        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.debug(f"x shape: {x.shape}")

        # Multi-head attention
        context, _ = self.multi_head_attention(x, x, x)
        logger.debug(f"context shape: {context.shape}")

        # Fully connected layer
        transformed_context = self.linear(context)
        transformed_context = self.tanh(transformed_context)
        logger.debug(f"transformed context shape: {transformed_context.shape}")

        # Additive attention
        additive_weights = self.additive_attention(context)

        # Weighted context by the attention weights
        out = torch.sum(additive_weights.unsqueeze(-1) * transformed_context, dim=1)
        logger.debug(f"user encoder out shape: {out.shape}")

        return out


class NRMS(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 768,
        encoder_dim: int = 128,
        num_heads_news_encoder: int = 8,
        num_heads_user_encoder: int = 8,
    ):
        super(NRMS, self).__init__()

        self.input_dim = input_dim
        self.encoder_dim = encoder_dim  # news & user encoder output dimension
        self.num_heads_news_encoder = num_heads_news_encoder
        self.num_heads_user_encoder = num_heads_user_encoder

        self.doc_encoder = DocEncoder()

        for param in self.doc_encoder.model.parameters():
            param.requires_grad = False

        for param in self.doc_encoder.model.encoder.layer[-1].parameters():
            param.requires_grad = True

        self.news_encoder = NewsEncoder(
            input_dim=input_dim,
            output_dim=encoder_dim,
            num_heads=num_heads_news_encoder,
        )
        self.user_encoder = UserEncoder(encoder_dim)
        self.criterion = nn.CrossEntropyLoss()

        self.training_step_outputs = []
        self.validating_step_outputs = []
        self.testing_step_outputs = []

        self.save_hyperparameters()

    def forward(
        self,
        clicked_ids: torch.Tensor,
        clicked_attention_mask: torch.Tensor,
        labeled_ids: torch.Tensor,
        labeled_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the NRMS model.

        Args:
            clicked_ids (torch.Tensor): Input token ids for the clicked news. (users, titles, seq_length)
            clicked_attention_mask (torch.Tensor): Attention mask for the clicked news. (users, titles, seq_length)
            labeled_ids (torch.Tensor): Input token ids for the labeled news. (users, titles, seq_length)
            labeled_attention_mask (torch.Tensor): Attention mask for the labeled news. (users, titles, seq_length)
        Returns:
            torch.Tensor: Scores for each candidate news. (users, 5)
            torch.Tensor: Candidate Context Weights.
            torch.Tensor: Candidate Additive Weights.
        """

        # labeled
        news_vectors, c_weights, a_weights = self.forward_news_encoder(
            labeled_ids, labeled_attention_mask
        )
        # shape: (users, 5, encoder_dim)

        # clicked
        user_vectors = self.forward_user_encoder(clicked_ids, clicked_attention_mask)
        # shape: (users, embed_size)

        # 각 뉴스와 사용자 벡터 간의 내적을 계산하여 scores 를 얻습니다.
        scores = torch.bmm(news_vectors, user_vectors.unsqueeze(2)).squeeze(
            2
        )  # shape: (users, 5)

        return scores, c_weights, a_weights

    def forward_doc_encoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass for the document encoder.

        Args:
            input_ids (torch.Tensor): Input token ids. (users, titles, seq_length)
            attention_mask (torch.Tensor): Attention mask. (users, titles, seq_length)

        Returns:
            torch.Tensor: Document embeddings. (users, titles, seq_length, embed_size)
        """
        users, titles, seq_length = input_ids.shape

        # reshape input_ids and attention_mask
        reshaped_input_ids = input_ids.view(users * titles, seq_length)
        reshaped_attention_mask = attention_mask.view(users * titles, seq_length)

        # forward
        embeddings = self.doc_encoder(reshaped_input_ids, reshaped_attention_mask)
        # rollback the shape and order
        embeddings = embeddings.view(users, titles, seq_length, self.input_dim)
        return embeddings  # shape: (users, titles, seq_length, embed_size)

    def forward_news_encoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass for the news encoder.

        Args:
            input_ids (torch.Tensor): Input token ids. (users, titles, seq_length)
            attention_mask (torch.Tensor): Attention mask. (users, titles, seq_length)

        Returns:
            torch.Tensor: News embeddings. (users, titles, encoder_dim)
        """

        embeddings = self.forward_doc_encoder(input_ids, attention_mask)
        users, titles, seq_length, embed_size = embeddings.shape

        embeddings = embeddings.view(users * titles, seq_length, embed_size)
        key_padding_mask = attention_mask.view(users * titles, seq_length)
        key_padding_mask = ~key_padding_mask.bool()

        news_vectors, c_weights, a_weights = self.news_encoder(embeddings, key_padding_mask)
        news_vectors = news_vectors.view(users, titles, self.encoder_dim)
        return news_vectors, c_weights, a_weights

    def forward_user_encoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass for the user encoder.

        Args:
            input_ids (torch.Tensor): Input token ids. (users, titles, seq_length)
            attention_mask (torch.Tensor): Attention mask. (users, titles, seq_length)

        Returns:
            torch.Tensor: User embeddings. (users, encoder_dim)
        """
        news_vectors, _, __ = self.forward_news_encoder(input_ids, attention_mask)
        users, titles, encoder_dim = news_vectors.shape

        # FORWARD HERE
        user_vector = self.user_encoder(news_vectors)

        # USER 가 읽은 기사가 모두 합쳐져서 벡터를 생성 하므로.. TITLE 개수 차원이 merge 되어야 함
        assert user_vector.shape == (users, encoder_dim)
        return user_vector

    def forward_and_compute_loss(self, batch: BatchEncoding):
        clicked_tokens, labeled_tokens, labels = batch

        scores, c_weights, a_weights = self.forward(
            clicked_ids=clicked_tokens["input_ids"],
            clicked_attention_mask=clicked_tokens["attention_mask"],
            labeled_ids=labeled_tokens["input_ids"],
            labeled_attention_mask=labeled_tokens["attention_mask"],
        )

        assert labels.shape == (scores.shape[0], 1)
        labels = labels.squeeze(1).long()

        loss = self.criterion(scores, labels)
        return loss, c_weights, a_weights

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-5)
        # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "avg_val_loss",
            # },
        }

    def training_step(self, batch: BatchEncoding, batch_idx: int):
        loss, _, __ = self.forward_and_compute_loss(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch: BatchEncoding, batch_idx: int):
        loss, c_weights, a_weights = self.forward_and_compute_loss(batch)

        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.validating_step_outputs.append(loss)

        if batch_idx % 3 == 0:
            # Attention visualization logging
            title = "Attention Weights"
            caption = f"{title} Batch-{batch_idx}"
            fig = utils.plot_2d_weights(weights=c_weights[0], title=title)
            wandb.log({"attention_weights": [wandb.Image(fig, caption=caption)]})

            # Additive softmax_results visualization logging
            title = "Additive Softmax"
            caption = f"{title} Batch-{batch_idx}"
            fig = utils.plot_2d_weights(weights=a_weights, title=title)
            wandb.log({"additive_softmax": [wandb.Image(fig, caption=caption)]})

        return {"val_loss": loss}

    def test_step(self, batch: BatchEncoding, batch_idx: int):
        loss, _, __ = self.forward_and_compute_loss(batch)

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
