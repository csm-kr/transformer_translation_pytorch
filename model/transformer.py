import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, source, target):
        # source = [batch size, source length]
        # target = [batch size, target length]
        encoder_output = self.encoder(source)                            # [batch size, source length, hidden dim]
        output, attn_map = self.decoder(target, source, encoder_output)  # [batch size, target length, output dim]
        return output, attn_map

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':

    import torch
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--feed_forward_dim', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--save_model', type=str, default='model.pt')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--optim', type=str, default='Adam')

    parser.add_argument('--input_dim', type=int, default=55002)
    parser.add_argument('--output_dim', type=int, default=19544)
    parser.add_argument('--sos_idx', type=int, default=2)
    parser.add_argument('--eos_idx', type=int, default=3)
    parser.add_argument('--pad_idx', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    opts = parser.parse_args()

    transformer_model = Transformer(opts).cuda()
    src = torch.randint(255, (10, 10),  dtype=torch.long).cuda()
    tgt = torch.randint(255, (10, 12), dtype=torch.long).cuda()

    out = transformer_model(src, tgt)
    print(out[0].size())
