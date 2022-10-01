import argparse
import random
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='SCOAD')

    parser.add_argument('--data_root', default='./data/Thumos14')
    parser.add_argument('--pretrained_ckpt', default='./data/checkpoint/thumos_iter_2300.pth')
    parser.add_argument('--max_iter', type=int, default=4000)
    parser.add_argument('--warm_start', type=int, default=0)
    parser.add_argument('--eval_intern', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--supervision', type=str, default='click')

    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_similar', type=int, default=3)

    parser.add_argument('--enc_step', type=int, default=64)
    parser.add_argument('--feature_size', type=int, default=2048)
    parser.add_argument('--hidden_size', type=int, default=4096)
    parser.add_argument('--num_class', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_seqlen', type=int, default=750)

    parser.add_argument('--mix_percent', type=float, default=0.9)
    parser.add_argument('--fps', type=float, default=1.5625)
    parser.add_argument('--sup_percent', type=float, default=1.0)

    args = parser.parse_args()

    return args


def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
