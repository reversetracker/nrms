import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, d_model: int, nhead: int):
        super(NewsEncoder, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(d_model, nhead)
        self.fc = nn.Linear(d_model, d_model)
        self.additional_attn = nn.Parameter(torch.randn(d_model))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        print(f"x shape: {x.shape}")
        # x shape: torch.Size([20, 8192, 128])

        attn_output, _ = self.multi_head_attention(x, x, x, key_padding_mask=key_padding_mask)

        print(f"attn_output shape: {attn_output.shape}")
        # attn_output shape: torch.Size([20, 8192, 128])

        # attention output 을 새로운 차원으로 projection 한다.
        # 데이터의 특징을 재조정 하고 필요한 정보를 과장 및 필요 없는 정보를 축소
        fc_output = self.fc(attn_output)
        print(f"fc_output shape: {fc_output.shape}")
        # fc_output shape: torch.Size([20, 8192, 128])

        # tanh 를 통해 데이터의 특징에 비선형성 추가.
        tanh_output = torch.tanh(fc_output)
        print(f"tanh_output shape: {tanh_output.shape}")
        # tanh_output shape: torch.Size([20, 8192, 128])

        # Additional Attention
        # word embedding 에 weight 을 곱해서 스칼라로 만듬 즉 각 단어의 중요도를 나타냄
        # 20 단어, 8192 (배치), 128 (임베딩 사이즈) -> 128 임베딩이 스칼라(중요도)로 변환 되며
        # 즉 20개의 단어들의 중요도를 scalar 로 표현
        additional_attn_output = tanh_output.matmul(self.additional_attn)
        print(f"additional_attn_output shape: {additional_attn_output.shape}")
        # additional_attn_output shape: torch.Size([20, 8192])

        # 각 단어의 중요도를 softmax 를 통해 확률로 만듬
        softmax_output = F.softmax(additional_attn_output, dim=1)
        print(f"softmax_output shape: {softmax_output.shape}")
        # softmax_output shape: torch.Size([20, 8192])

        # shape 체크.
        print("softmax_output.unsqueeze(-1) shape:", softmax_output.unsqueeze(-1).shape)
        # softmax_output.unsqueeze(-1) shape: torch.Size([20, 8192, 1])
        print("attn_output shape:", attn_output.shape)
        # attn_output shape: torch.Size([20, 8192, 128])

        # Aggregate the Output
        # 각단어의 중요도를 attention_output 에 곱해서 각 단어의 중요도에 따라서
        # attention_output 을 조정
        out = torch.sum(softmax_output.unsqueeze(-1) * attn_output, dim=0)
        print(f"out shape: {out.shape}")
        # out shape: torch.Size([8192, 128])
        return out


class UserEncoder(nn.Module):
    def __init__(self, d_model, nhead):
        super(UserEncoder, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(d_model, nhead)
        self.fc = nn.Linear(d_model, d_model)
        self.additional_attn = nn.Parameter(torch.randn(d_model))

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

        # 각 기사 벡터를 중요도 scalar로 변환 하기 위해 query vector 연산.
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
        out = torch.sum(weighted_attention, dim=1)
        print(f"out shape: {out.shape}")
        return out
