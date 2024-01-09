import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch
from recomand import Rating_net, ALS_dataset

a=pd.read_csv("./data/movies.csv")
b=pd.read_csv("./data/tags.csv")
c=pd.read_csv("./data/ratings.csv")
d=pd.read_csv("./data/genome-tags.csv")
e=pd.read_csv("./data/genome-scores.csv")
pass