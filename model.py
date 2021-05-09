import torch
import math
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_head=8):
        super().__init__()
        self.num_head = num_head
        self.model_dim = model_dim
        self.d_k = self.d_v = model_dim // num_head  # 512 / 8 = 64

        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        self.W_o = nn.Linear(model_dim, model_dim)

    def self_attention(self, q, k, v, mask=None):
        """

        Args:
            q:     query [B, token, 512]
            k:     key   [B, token, 512]
            v:     value [B, token, 512]
            mask:  mask  [B, 1, 1, 512]

        Returns: salced dot product results [B, 100, 512]

        """
        # 1. split as head      [b, TOKEN 512], -> [b, token, HEAD, 64] -> [B, num_head, token, 64]
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.num_head, self.d_k).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_head, self.d_k).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_head, self.d_k).permute(0, 2, 1, 3)

        # 2. q * k^t (transpose of k) & scaled dot product
        qk = torch.matmul(q, k.permute(0, 1, 3, 2))                                   # [B, num_head, token, token]
        attention_score = qk / math.sqrt(self.d_k)

        # softmax 앞에서 masking (정보가 없는 부분의 score 를 매우 작은 값으로 만듦)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e20)

        softmax_attention_score = torch.softmax(attention_score, dim=-1)              # [B, num_head, token, token]
        result = torch.matmul(softmax_attention_score, v)                             # [B, num_head, token, dk]

        # concat the multi-head result semantically
        result = result.permute(0, 2, 1, 3).reshape(batch_size, -1, self.model_dim)   # [B, token, h, dk -> B, 100, 512]
        return result

    def forward(self, x, mask=None):
        # x shape is                                     [batch, num_token, model_dim] - [4, 100, 512]
        q = self.W_q(x)    # [B, 100, 512]
        k = self.W_k(x)    # [B, 100, 512]
        v = self.W_k(x)    # [B, 100, 512]

        if mask is not None:
            mask = mask.unsqueeze(1)                    # [B, 1, 1, 100] for broadcasting.

        attention = self.self_attention(q, k, v, mask)  # [B, 100, 512]
        x = self.W_o(attention)                         # [B, 100, 512]
        return x


class FeedForwardNet(nn.Module):
    def __init__(self, model_dim, dropout):
        super().__init__()

        self.ffd = nn.Sequential(nn.Linear(model_dim, model_dim * 4),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(model_dim * 4, model_dim))

    def forward(self, x, mask):
        if mask is None:
            return self.ffd(x)


class ResidualConnection(nn.Module):
    def __init__(self, model_dim, dropout):
        super(ResidualConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, module, mask=None):
        """

        Args:
            x:
            module: MultiHeadAttention or FeedForwardNet

        Returns: layer_norm(residual(dropout(module(X, mask))))

        """
        residual = x
        x = module(x, mask)
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        # post LayerNorm original version of transformer but there is pre LayerNorm
        return x


class EncoderBlock(nn.Module):
    def __init__(self, model_dim, num_head, dropout=0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(model_dim=model_dim, num_head=num_head)
        self.residual_1 = ResidualConnection(model_dim=model_dim, dropout=dropout)

        self.feed_forward = FeedForwardNet(model_dim=model_dim, dropout=dropout)
        self.residual_2 = ResidualConnection(model_dim=model_dim, dropout=dropout)

    def forward(self, x, mask):
        x = self.residual_1(x, module=self.multi_head_attention, mask=mask)
        x = self.residual_2(x, module=self.multi_head_attention)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_head, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList([])

        for _ in range(self.num_layers):
            self.encoder_layers.append(EncoderBlock(model_dim, num_head, dropout))

    def forward(self, x, mask=None):
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x, mask)
        return x


if __name__ == '__main__':
    num_layers, model_dim, num_head, dropout = 6, 512, 8, 0.1
    model = Encoder(num_layers, model_dim, num_head, dropout)
    tensor = torch.randn([4, 100, 512])
    mask = torch.ones([4, 1, 100])
    print(model(tensor, mask).size())
