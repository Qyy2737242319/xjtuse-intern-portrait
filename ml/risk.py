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

class Risk_network(nn.Module):
    def __init__(self):
        super(Risk_network, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32),
            nn.ReLU(inplace=True)
        )
