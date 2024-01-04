import gc

import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import tqdm
import os
import argparse
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


class ALS_net(nn.Module):

    def __init__(self, user_num, goods_num, k):
        super(ALS_net, self).__init__()
        if os.path.exists("./weights/user_matrix_latest.npy"):
            self.user_matrix = torch.tensor(np.load("./weights/user_matrix_latest.npy"), dtype=torch.float32).cuda()
            self.goods_matrix = torch.tensor(np.load("./weights/goods_matrix_latest.npy"), dtype=torch.float32).cuda()
        else:
            self.user_matrix = torch.rand((user_num, k), dtype=torch.float32).cuda()
            self.goods_matrix = torch.rand((k, goods_num), dtype=torch.float32).cuda()
        self.user_matrix.requires_grad = True
        self.goods_matrix.requires_grad = True

    def forward(self, location):
        x = torch.dot(self.user_matrix[location[0, 0], ...], self.goods_matrix[..., location[0, 1]]).unsqueeze(-1)
        for i in range(1, location.shape[0]):
            x = torch.concatenate((x, torch.dot(self.user_matrix[location[i, 0], ...],
                                                self.goods_matrix[..., location[i, 1]]).unsqueeze(-1)), dim=0)
        return x.unsqueeze(dim=1)


class ALS_dataset(torch.utils.data.Dataset):
    def __init__(self, name, file_path):
        self.name = name
        # self.random_seed = random_seed
        self.data = torch.Tensor(pd.read_csv(file_path).values[..., :-1]).permute(1, 0)  # user_id, movie_id, rating
        self.user_num = int(torch.max(self.data[0, ...]).item())
        self.goods_num = int(torch.max(self.data[1, ...]).item())

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        x = int(self.data[0][idx].item())
        y = int(self.data[1][idx].item())
        return self.data[2][idx] / 5, torch.tensor([x - 1, y - 1], dtype=torch.int32)

    def get_size(self):
        return self.user_num, self.goods_num


def train(args):
    train_dataset = ALS_dataset('ALS_train.csv', 'data/ratings.csv')
    loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8)
    writter = SummaryWriter('./logs')
    user_num, goods_num = train_dataset.get_size()
    model = ALS_net(user_num, goods_num, k=args.k_value)
    optimizer = torch.optim.Adam([model.user_matrix, model.goods_matrix], lr=args.lr)
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    model.cuda()

    for step, data in tqdm.tqdm(enumerate(loader)):
        step = step + args.resume
        optimizer.zero_grad()
        true_data, idx = data
        true_data = true_data.reshape(-1, 1).cuda()
        fake_data = model.forward(idx)
        loss = criterion(fake_data, true_data)+criterion2(fake_data, true_data)
        loss.backward()
        optimizer.step()

        if step != 0 and step % args.logs_iter == 0:
            writter.add_scalar("loss", loss, step)
            writter.flush()
            print("loss:" + str(loss.item()))
            np.save(f"./weights/user_matrix_latest.npy", model.user_matrix.detach().cpu().numpy())
            np.save(f"./weights/goods_matrix_latest.npy", model.goods_matrix.detach().cpu().numpy())
            gc.collect()

        if step != 0 and step % args.save_iter == 0:
            np.save(f"./weights/user_matrix_{step}.npy", model.user_matrix.detach().cpu().numpy())
            np.save(f"./weights/goods_matrix_{step}.npy", model.goods_matrix.detach().cpu().numpy())

        if step == args.total_step:
            print("training complete!")
            break


parser = argparse.ArgumentParser()
parser.add_argument("--k_value", help="the transition rank value of matrix", default=10, required=False)
parser.add_argument("--batch_size", help="the batch size of training", default=50000, required=False)
parser.add_argument("--total_step", help="the total number of training steps", default=10000, required=False)
parser.add_argument("--lr", help="the learning rate of training", default=0.0001, required=False)
parser.add_argument("--logs_iter", help="log after how many iterations", default=4, required=False)
parser.add_argument("--save_iter", help="save after how many iterations", default=50, required=False)
parser.add_argument("--resume", help="save after how many iterations", default=0, required=False)
args = parser.parse_args()
train(args)
