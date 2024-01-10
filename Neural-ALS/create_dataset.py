import gc
import pandas as pd
import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--user_amount", help="the batch size of training", default=1000, required=False)
parser.add_argument("--goods_amount", help="the epoch number of training", default=1000, required=False)
parser.add_argument("--p_goods_low", help="the learning rate of training", default=0.6, required=False)
parser.add_argument("--p_user_low", help="the learning rate of training", default=0.5, required=False)

parser.add_argument("--loss_rate", help="the learning rate of training", default=0.6, required=False)

args = parser.parse_args()


def generate():
    main_matrix = torch.ones((args.user_amount, args.goods_amount), dtype=torch.float32) * -1
    value = [i / 2 for i in range(11)]
    low_value_p = [0.05, 0.09, 0.11, 0.15, 0.18, 0.15, 0.11, 0.09, 0.05, 0.02, 0]
    high_value_p = [0, 0.02, 0.05, 0.09, 0.11, 0.15, 0.18, 0.15, 0.11, 0.09, 0.05]
    low_user_amount = int(args.user_amount * args.p_user_low)
    low_goods_amount = int(args.goods_amount * args.p_goods_low)
    low_user_index = torch.randint(1, args.user_amount, (low_user_amount, 1))
    low_goods_index = torch.randint(1, args.goods_amount, (low_goods_amount,))
    main_matrix[low_user_index, low_goods_index] = torch.tensor(
        np.random.choice(value, [low_user_amount, low_goods_amount], p=low_value_p), dtype=torch.float32)
    index = torch.nonzero(
        torch.where(main_matrix < 0, torch.ones(main_matrix.size()), torch.zeros(main_matrix.size())))
    high_value_matrix = torch.tensor(
        np.random.choice(value, [main_matrix.shape[0], main_matrix.shape[1]], p=high_value_p), dtype=torch.float32)
    main_matrix_ = torch.where(main_matrix < 0, high_value_matrix, main_matrix)
    # main_matrix[index[..., 0], index[..., 1]] = torch.tensor(
    #     np.random.choice(value, [index.shape[0], 1], p=high_value_p), dtype=torch.float32)

    loss_user_amount = int(args.loss_rate * args.user_amount)
    loss_goods_amount = int(args.loss_rate * args.goods_amount)
    loss_user_index = torch.randint(1, args.user_amount, (loss_user_amount, 1))
    loss_goods_index = torch.randint(1, args.goods_amount, (loss_goods_amount,))
    main_matrix_[loss_user_index, loss_goods_index] = -3
    index = torch.nonzero(
        torch.where(main_matrix_ >= 0, torch.ones(main_matrix_.size()), torch.zeros(main_matrix_.size())))
    df = pd.DataFrame(
        {'user_id': index[..., 0].numpy(), 'goods_id': index[..., 1].numpy(),
         'rate': main_matrix_[index[..., 0], index[..., 1]].numpy().reshape(-1)})
    print("writing...")
    df.to_csv('./output.csv', index=False)

    return 0


generate()
