import gc
import pandas as pd
import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Rating_net(nn.Module):

    def __init__(self):
        super(Rating_net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(51, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 11)
        )

    def forward(self, x):
        y = self.net(x)
        return y


class Goods_space(nn.Module):

    def __init__(self):
        super(Goods_space, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1129, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 50)
        )

    def forward(self, x):
        y = self.net(x)
        return y


class Instant_ALS_net(nn.Module):

    def __init__(self, num_users, num_items, k, dev):
        super(Instant_ALS_net, self).__init__()
        self.dev0 = dev[0]
        self.dev1 = dev[1]
        self.dev2 = dev[2]
        self.k = k
        self.encoder1 = nn.Embedding(num_users, k).to(self.dev0)
        self.encoder2 = nn.Embedding(num_items, k).to(self.dev1)
        self.net = nn.Sequential(
            # nn.Conv1d(2, 4, 5, 1),
            # nn.MaxPool1d(5, 2),
            # nn.Conv1d(4, 6, 5, 1),
            # nn.MaxPool1d(5, 2),
            # nn.Flatten(),
            nn.Linear(2 * k, 8000),
            nn.ReLU(),
            nn.Linear(8000, 4000),
            nn.ReLU(),
            nn.Linear(4000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1)
        ).to(self.dev2)

    def forward(self, user, item):
        user = user.to(self.dev0)
        item = item.to(self.dev1)
        user_latent = self.encoder1(user).reshape(-1, self.k).to(self.dev2)
        item_latent = self.encoder2(item).reshape(-1, self.k).to(self.dev2)
        latent = torch.cat([user_latent, item_latent], dim=1)
        return self.net(latent)


class Instant_ALS_light_net(nn.Module):

    def __init__(self, num_users, num_items, k, dev0, dev1):
        super(Instant_ALS_light_net, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.encoder1 = nn.Embedding(num_users, k).to(dev0)
        self.encoder2 = nn.Embedding(num_items, k).to(dev0)
        self.net = nn.Sequential(
            nn.Linear(2 * k, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1),
        ).to(dev1)

    def forward(self, user, item):
        user = user.to(self.dev0)
        item = item.to(self.dev0)
        user_latent = self.encoder1(user).squeeze()
        item_latent = self.encoder2(item).squeeze()
        latent = torch.cat([user_latent, item_latent], dim=1).to(self.dev1)
        return self.net(latent)


class ALS_dataset(torch.utils.data.Dataset):
    def __init__(self, name, rating_file_path, tag_file_path):
        self.name = name
        # self.random_seed = random_seed
        self.rating_data = torch.Tensor(pd.read_csv(rating_file_path).values).permute(1, 0)  # user_id, movie_id, rating
        # self.tags_data = torch.Tensor(pd.read_csv(tag_file_path).values).permute(1, 0).reshape(3, -1, 1128)
        self.user_num = int(torch.max(self.rating_data[0, ...]).item())
        self.goods_num = int(torch.max(self.rating_data[1, ...]).item())

    def __len__(self):
        return self.rating_data.shape[1]

    def __getitem__(self, idx):
        # tags = self.tags_data[2, idx, :].reshape(-1)
        # a = torch.nonzero(self.rating_data[1, ...] == int(self.tags_data[0, idx, 0].squeeze().item())).squeeze()
        # randnum = torch.randint(0, a.shape[0] - 1, [1, ])
        user_id = int(self.rating_data[0, idx].item())
        goods_id = int(self.rating_data[1, idx].item())
        # a = torch.nonzero(self.tags_data[0, :, 0] == goods_id).squeeze()
        # tags = self.tags_data[2, a, :]
        # tags = torch.tensor(tags[(tags[0, :, 0] == goods_id).nonzero().squeeze(1)], dtype=torch.float32)
        # for i in range(self.tags_data.shape[1]):
        #     if int(self.tags_data[0, i, 0].item()) == goods_id:
        #         tags = self.tags_data[2, i, ...]
        #         break
        rating = self.rating_data[2, [idx]] / 5
        # rate = torch.zeros([11])
        # i = int(rating / 0.5)
        # rate[i] = 1

        return torch.LongTensor([user_id - 1]), torch.LongTensor(
            [goods_id - 1]), rating


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '13355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def train(rank, world_size):
    setup(rank, world_size)
    dev = [0, 1, 2]
    train_dataset = ALS_dataset('ALS_train.csv', './data/ratings.csv', "./data/genome-scores.csv")
    model = Instant_ALS_net(train_dataset.user_num, train_dataset.goods_num, args.k, dev)
    ddp_model = DDP(model)

    if os.path.exists("./weights/model_latest.pt"):
        ddp_model.load_state_dict(torch.load("./weights/model_latest.pt"))
        dist.barrier()

    loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=20)
    writter = SummaryWriter('./logs')

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.lr)
    # optimizer2 = torch.optim.SparseAdam(model.encoder1.parameters(), lr=param.lr)
    # optimizer3 = torch.optim.SparseAdam(model.encoder2.parameters(), lr=param.lr)
    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()
    criterion3 = nn.CrossEntropyLoss()
    # model.cuda()
    torch.cuda.empty_cache()
    gc.collect()
    for epoch in range(args.epoch):
        for step, data in enumerate(tqdm.tqdm(loader)):
            step = step + args.resume + int(epoch * train_dataset.__len__() / args.batch_size)
            optimizer.zero_grad()
            # optimizer2.zero_grad()
            # optimizer3.zero_grad()
            user_id, goods_id, ratings = data
            # tags = tags.cuda()
            ratings = ratings.to(dev[2])
            fake_data = ddp_model(user_id, goods_id)
            loss = 5 * (criterion(fake_data, ratings) + criterion2(fake_data, ratings))
            # loss = criterion3(fake_data, ratings)
            loss.backward()
            optimizer.step()
            # optimizer2.step()
            # optimizer3.step()

            if step != 0 and step % args.logs_iter == 0:
                writter.add_scalar("net_loss", loss, step)
                writter.flush()
                print("loss:" + str(loss.item()))
                # print(model.encoder1.parameters())

            if step != 0 and step % args.save_iter == 0:
                torch.cuda.empty_cache()
                gc.collect()
                # torch.save(model.state_dict(), f'./weights/model_latest.pt')
                if rank == 0:
                    # All processes should see same parameters as they all start from same
                    # random parameters and gradients are synchronized in backward passes.
                    # Therefore, saving it in one process is sufficient.
                    torch.save(ddp_model.state_dict(), f'./weights/model_latest.pt')
                    dist.barrier()
                    ddp_model.eval()
                    evaluate(train_dataset, ddp_model, dev)
                    ddp_model.train()


def evaluate(train_dataset, model, dev):
    pass
    select = torch.randint(0, train_dataset.__len__(), [args.eval_num])
    eval_user_id = (train_dataset.rating_data[0, select] - 1).long().unsqueeze(dim=1)
    eval_goods_id = torch.LongTensor((train_dataset.rating_data[1, select] - 1).long().unsqueeze(dim=1))
    rating = train_dataset.rating_data[2, select].unsqueeze(dim=1).to(dev[2])
    # tags = train_dataset.tags_data

    # eval_input = torch.cat((eval_user_id, eval_goods_id), dim=1)
    fake_data = model(eval_user_id, eval_goods_id)
    # fake_rating = torch.zeros((args.eval_num, 1)).to(dev1)
    # for i in range(args.eval_num):
    #     _, a = torch.topk(fake_data[i, ...], k=1)
    #     fake_rating[i, 0] = a * 0.5
    fake_rating = torch.clamp(torch.round(fake_data * 5 * 2) / 2, 0, 5)
    accurate = 0
    blur_accurate = 0
    for i in tqdm.tqdm(range(fake_rating.shape[0]), desc="evaluating...", colour='red'):
        if rating[i].item() - 0.5 <= fake_rating[i].item() <= rating[i].item() + 0.5:
            blur_accurate += 1
        if torch.equal(fake_rating[i], rating[i]):
            accurate += 1

    print("accuracy: " + str(accurate / args.eval_num))
    print("blur_accuracy: " + str(blur_accurate / args.eval_num))


def test(rank, world_size):
    setup(rank, world_size)
    dev = [0, 1, 2]

    train_dataset = ALS_dataset('ALS_train.csv', './data/ratings.csv', "./data/genome-scores.csv")
    model = Instant_ALS_net(train_dataset.user_num, train_dataset.goods_num, args.k, dev)
    ddp_model = DDP(model)
    if os.path.exists("./weights/model_latest.pt"):
        ddp_model.load_state_dict(torch.load("./weights/model_latest.pt"))
    ddp_model.eval()
    user_id = torch.LongTensor([[6886 - 1]]).to(dev[0])
    goods_id = torch.LongTensor([[8235 - 1]]).to(dev[1])
    predict = ddp_model(user_id, goods_id)
    predict = torch.clamp(torch.round(predict * 5 * 2) / 2, 0, 5)
    print("the rating predicted for user_id:" + str(user_id) + " and goods_id" + str(goods_id) + "is " + str(predict))


def run(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="the batch size of training", default=131072, required=False)  # 131072
parser.add_argument("--epoch", help="the epoch number of training", default=10000, required=False)
parser.add_argument("--lr", help="the learning rate of training", default=0.000005, required=False)
parser.add_argument("--logs_iter", help="log after how many iterations", default=20, required=False)
parser.add_argument("--save_iter", help="save after how many iterations", default=100, required=False)
parser.add_argument("--resume", help="resume training from iteration number", default=6901, required=False)
parser.add_argument("--eval_num", help="evaluation number for each save iter", default=131072, required=False)
parser.add_argument("--k", help="feature channel of embeddings", default=4200, required=False)

args = parser.parse_args()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus // 2
    run(train, world_size)
    #run(test, world_size)
