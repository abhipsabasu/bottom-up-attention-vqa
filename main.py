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
import sys

from vqa_debias_loss_functions import *

parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
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
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    cp = not args.nocp
    print('gsdjgsjghshg')
    print("Building train dataset...")
    train_dset = VQAFeatureDataset('dev', dictionary, cp=cp,
                                   cache_image_features=args.cache_features)
    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('test', dictionary, cp=cp,
                                  cache_image_features=args.cache_features)

    answer_voc_size = train_dset.num_ans_candidates

    # Compute the bias:
    # The bias here is just the expected score for each answer/question type

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)
    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.entries:
        ans = ex["answer"]
        q_type = ans["question_type"]
        question_type_to_count[q_type] += 1
        if ans["labels"] is not None:
            for label, score in zip(ans["labels"], ans["scores"]):
                question_type_to_probs[q_type][label] += score

    question_type_to_prob_array = {}
    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        # prob_array /= count
        question_type_to_prob_array[q_type] = prob_array
    # count_arr_train = [0] * 2274
    # for q_type, stat in question_type_to_prob_array.items():
    #     for i in range(2274):
    #         count_arr_train[i] = count_arr_train[i] + stat[i]
    # count_arr_train.sort()
    # count_idx = np.argsort(count_arr_train)
    # count_idx = count_idx[::-1]
    #
    # plt.semilogy([i for i in range(2274)], count_arr, label='Test Label Distribution')
    # plt.semilogy([i for i in range(2274)], count_arr_train, label='Training Label Distribution')
    # plt.title('Label Distribution for The dataset - overall')
    # plt.legend()
    # plt.savefig('label_distribution.png')
    # print(count_arr[-10:])
    # print('Reached...#########')
    # Now add a `bias` field to each example
    for ds in [train_dset, eval_dset]:
        for ex in ds.entries:
            q_type = ex["answer"]["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model, basemodel = getattr(base_model, constructor)(train_dset, args.num_hid)


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
    # basemodel = basemodel.cuda()
    model = model.cuda()
    model.apply(weights_init_kn)
    batch_size = args.batch_size


    # The original version uses multiple workers, but that just seems slower on my setup
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Starting training...")
    train(model, basemodel, train_loader, eval_loader, args.epochs, args.output, True, seed, 'original')
    # train(model, reconstruction_model, train_loader, eval_loader, args.epochs // 3, args.output, False, seed,
    #       'reconstruction')
    # train(model, reconstruction_model, train_loader, eval_loader, args.epochs, args.output, True, seed,
    #       'combination')
#
#
if __name__ == '__main__':
    main()