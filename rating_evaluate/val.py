import torch
import numpy as np
import os
import tqdm
import pandas as pd
import tqdm

user_matrix = torch.tensor(np.load("./weights_save/user_matrix_latest.npy"), dtype=torch.float32).cuda()
goods_matrix = torch.tensor(np.load("./weights_save/goods_matrix_latest.npy"), dtype=torch.float32).cuda()


def val(location):
    x = torch.dot(user_matrix[location[0, 0], ...], goods_matrix[..., location[0, 1]]).unsqueeze(-1)
    for i in range(1, location.shape[0]):
        x = torch.concatenate((x, torch.dot(user_matrix[location[i, 0], ...],
                                            goods_matrix[..., location[i, 1]]).unsqueeze(-1)), dim=0)
    print("before revision: " + str(x * 5))
    print("after revision: " + str(torch.clamp(torch.round(x * 5 * 2) / 2, 0, 5)))


def test():
    data = torch.Tensor(pd.read_csv("./data/ratings.csv").values[..., :-1]).permute(1, 0)
    location = data[:2, :500000].permute(1, 0)
    x = torch.dot(user_matrix[int(location[0, 0].item() - 1), ...],
                  goods_matrix[..., int(location[0, 1].item() - 1)]).unsqueeze(-1)
    for i in range(1, location.shape[0]):
        x = torch.concatenate((x, torch.dot(user_matrix[int(location[i, 0].item() - 1), ...],
                                            goods_matrix[..., int(location[i, 1].item() - 1)]).unsqueeze(-1)), dim=0)
    x = x.unsqueeze(dim=1)
    true = data[[2], :500000].permute(1, 0)
    loss = 0
    for i in range(true.shape[0]):
        loss += (torch.abs(true[i, 0] - x[i, 0] * 5) / true[i, 0]).item()
    loss /= true.shape[0]
    print("before revision accuracy: " + str(1 - loss))
    x = torch.clamp(torch.round(x * 5 * 2) / 2, 0, 5)
    loss = 0
    for i in range(true.shape[0]):
        loss += (torch.abs(true[i, 0] - x[i, 0]) / true[i, 0]).item()
    loss /= true.shape[0]
    print("after revision accuracy: " + str(1 - loss))


test()
location = torch.tensor([[7044, 3995]], dtype=torch.int32).cuda()
val(location)
