import torch.nn as nn
import torch

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))

# 10  : data 길이 (length)
# 32  : batch
# 512 : d_model (dimension of model)

out = transformer_model(src, tgt)
print(out.size())