import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        dim_embed: int = 512,
        num_max_sequence: int = 5000,
        scale: int = 10000,
        dropout: float = 0,
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings in log space
        pe = torch.zeros(num_max_sequence, dim_embed)
        position = torch.arange(0, num_max_sequence).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_embed, 2) * -(torch.log(torch.Tensor([scale])) / dim_embed)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if dim_embed % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PositionalMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_max_sequence: int = 5000,
        scale: int = 10000,
        dropout: float = 0,
        batch_first: bool = True,
        *args,
        **kwargs,
    ):
        super(PositionalMultiheadAttention, self).__init__(*args, **kwargs)

        self.pe = PositionalEncoding(
            embed_dim, num_max_sequence=num_max_sequence, scale=scale, dropout=dropout
        )
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=batch_first
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        use_positional_encoding: bool = True,
    ):
        if use_positional_encoding:
            x = self.pe(x)

        context, context_weights = self.multi_head_attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        return context, context_weights


class EmblemNet(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_aa: int,
        dropout_p: float,
        num_heads: int = 1,
    ):
        super(EmblemNet, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embed_dim, padding_idx=0)

        self.ref_attention = PositionalMultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_max_sequence=201, dropout=dropout_p
        )
        self.alt_attention = PositionalMultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_max_sequence=201, dropout=dropout_p
        )
        self.msa_attention = PositionalMultiheadAttention(
            embed_dim=n_aa, num_heads=num_heads, num_max_sequence=201, dropout=dropout_p
        )

        self.ref_fc = nn.Linear(embed_dim, embed_dim)
        self.alt_fc = nn.Linear(embed_dim, embed_dim)
        self.msa_fc = nn.Linear(n_aa, n_aa)

        self.tanh = nn.Tanh()

        self.ref_lstm = nn.LSTM(embed_dim + n_aa, output_dim, batch_first=True)
        self.alt_lstm = nn.LSTM(embed_dim + n_aa, output_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, ref_tensor, alt_tensor, msa_tensor):
        x_ref = self.embedding(ref_tensor)
        x_ref, _ = self.ref_attention(x_ref)
        x_ref = self.ref_fc(x_ref)
        x_ref = self.tanh(x_ref)

        x_alt = self.embedding(alt_tensor)
        x_alt, _ = self.alt_attention(x_alt)
        x_alt = self.alt_fc(x_alt)
        x_alt = self.tanh(x_alt)

        x_msa, _ = self.msa_attention(msa_tensor)
        x_msa = self.msa_fc(x_msa)
        x_msa = self.tanh(x_msa)

        x_ref = torch.cat((x_ref, x_msa), dim=2)
        x_alt = torch.cat((x_alt, x_msa), dim=2)

        x_ref, _ = self.ref_lstm(x_ref)
        x_alt, _ = self.alt_lstm(x_alt)

        x_ref = x_ref[:, -1, :]
        x_alt = x_alt[:, -1, :]

        x_ref = self.dropout(x_ref)
        x_alt = self.dropout(x_alt)

        x = torch.cat((x_ref, x_alt), dim=1)
        return x
