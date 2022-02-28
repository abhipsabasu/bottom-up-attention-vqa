import argparse
import json
import pickle
from collections import defaultdict, Counter
from os.path import dirname, join

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from dataset import Dictionary, VQAFeatureDataset, VQAShuffleDataset
import base_model
from train import train
import utils
import random
import sys

from vqa_debias_loss_functions import *

parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = True


def weights_init_kn(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0.01)


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument(
        '--cache_features', action="store_true",
        help="Cache image features in RAM. Makes things much faster, "
             "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument(
        '--nocp', action="store_true", help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument(
        '-p', "--entropy_penalty", default=0.36, type=float,
        help="Entropy regularizer weight for the learned_mixin model")
    parser.add_argument(
        '--mode', default="learned_mixin",
        choices=["learned_mixin", "reweight", "bias_product", "none"],
        help="Kind of ensemble loss to use")
    parser.add_argument(
        '--eval_each_epoch', action="store_true",
        help="Evaluate every epoch, instead of at the end")

    # Arguments from the original model, we leave this default, except we
    # set --epochs to 15 since the model maxes out its performance on VQA 2.0 well before then
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    cp = not args.nocp

    print("Building train dataset...")
    train_dset = VQAFeatureDataset('train', dictionary, cp=cp,
                                   cache_image_features=args.cache_features)


    # Build the model using the original constructor
    constructor = 'build_pretrain'
    model, basemodel = getattr(base_model, constructor)(train_dset, args.num_hid)
    # basemodel.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    # Add the loss_fn based our arguments
    if args.mode == "bias_product":
        model.debias_loss_fn = BiasProduct()
    elif args.mode == "none":
        model.debias_loss_fn = Plain()
    elif args.mode == "reweight":
        model.debias_loss_fn = ReweightByInvBias()
    elif args.mode == "learned_mixin":
        model.debias_loss_fn = LearnedMixin(args.entropy_penalty)
    else:
        raise RuntimeError(args.mode)
    # constructor = 'build_reconstruction'
    # reconstruction_model = getattr(base_model, constructor)(args.num_hid).cuda()
    # Record the bias function we are using
    utils.create_dir(args.output)
    with open(args.output + "/debias_objective.json", "w") as f:
        js = model.debias_loss_fn.to_json()
        json.dump(js, f, indent=2)

    model = model.cuda()
    model.apply(weights_init_kn)
    batch_size = args.batch_size


    # The original version uses multiple workers, but that just seems slower on my setup
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Starting training...")
    train(model, None, train_loader, None, 15, args.output, False, seed, 'original')


if __name__ == '__main__':
    main()