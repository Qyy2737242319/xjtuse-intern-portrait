import gc
import pandas as pd
import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Instant_ALS_net(nn.Module):

    def __init__(self, num_users, num_items, k, dev0, dev1):
        super(Instant_ALS_net, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.encoder1 = nn.Embedding(num_users, k)
        self.encoder2 = nn.Embedding(num_items, k)
        self.net1 = nn.Conv1d(2, 4, 5, 1)
        self.net2 = nn.MaxPool1d(5, 2)
        self.net3 = nn.Conv1d(4, 6, 5, 1)
        self.net4 = nn.MaxPool1d(5, 2)
        self.net5 = nn.Flatten()
        self.net6 = nn.Linear(5 * 1993, 2000)
        nn.ReLU(),
        nn.Linear(2000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1),

    def forward(self, user, item):
        input = torch.cat((user, item), dim=0).unsqueeze(dim=0)
        x = self.net1(input)
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)
        x = self.net5(x)
        pass
        return


dev0 = 0
dev1 = 1
a = Instant_ALS_net(20000, 20000, 2000, dev0, dev1)
user = torch.zeros((1, 2000))
item = torch.zeros((1, 2000))
a(user, item)
