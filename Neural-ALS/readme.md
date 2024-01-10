<p align="center">
    <a href="https://github.com/Qyy2737242319/xjtuse-intern-portrait/edit/main/rating_evaluate">
      <img src="https://img.shields.io/badge/Neural--ALS-1.0.0-brightgreen" alt="neural-ALS">
    </a>
    <a href="#">
        <img src="https://img.shields.io/github/license/huccct/vue-admin">
    </a>
</p>

This repository demonstrates Neural-ALS, an optimized ALS algorithm based on deep learning for the dataset [ml-25m](https://grouplens.org/datasets/movielens/25m/) which conotains about 25,000,000 ratings of various movies from different viewers.

The algorithm shows great accuracy during training and we also provide a light network for training speed accelaration and model testing, when accuracy still close to the main model.


## Train Result

<img src="./train_results/net_loss.png" width=auto height=250> </img>

## Quick Start

To have a test of our model ability, run bash command like:
```
python ./recommand_predict.py --mode test --test_data [[1,1],[234,436],[4295,13895]]
```
where test_data value should be replaced by batch of user_id and movies_id you interested in.


## Environment Preparation

Creating environment using bash command below:

```
conda env create -f requirements.yml
pip install -r requirements.txt
```

## Device Requirements & Training Detail

* We deployed parallel computation on 3 nvidia rtx4090 gpus for training and evaluating stages and 20 cpu cores for data loading, and the super parameters we used had set default in the scripts.
* If you don't meet the requirements, you need to revise the code and some super parameters to enjoy the model.
* In our experiments, Neural-ALS needs about 9hrs to reduce the loss to 0.1 and 15hrs to reach convergence.
* During our experiments, we found that the super parameter k is proportional to the final accuracy, so increase the value of k as large as you can to enjoy the maximum performance.

## Training Settings

Train a Neural-ALS model via:

```
python ./recommand_predict.py --mode train [optional: --k --batch_size --epoch --lr --log_iter --save_iter --resume]
```

## Optional Tools
we provide some useful tools for surveillance listed here.

### using tensorboard to supervise training data

```
tensorboard --logdir ./logs
```
### using matlpotlib to supervise training data

```
python ./tools/matloss.py ./logs
```
