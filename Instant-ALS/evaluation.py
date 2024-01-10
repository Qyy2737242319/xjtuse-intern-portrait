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
from recomand import Rating_net, evaluate

model = Rating_net()
model.load_state_dict(torch.load("./weights/model_latest.pt"))
model.cuda()
model.eval()
input = torch.tensor([[136/5999, 4748/5999]], dtype=torch.float32).cuda()
result = model(input)
print(result*5)
result = torch.clamp(torch.round(result * 5 * 2) / 2, 0, 5)
print(result)
