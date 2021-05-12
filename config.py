import torch
import argparse

device_ids = [0]
device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')


def parse(args):
    # 1. arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50)                  # 14 / 20
    parser.add_argument('--port', type=str, default='8097')
    parser.add_argument('--lr', type=float, default=1e-3)                 # 1e-2
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)       # 0.0001
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--warm_up', type=int, default=4000)

    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='kor-eng')                         # FIXME

    parser.add_argument('--conf_thres', type=float, default=0.05)
    parser.add_argument('--start_epoch', type=int, default=0)

    opts = parser.parse_args(args)
    return opts