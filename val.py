import argparse
import json
import pickle
from collections import defaultdict, Counter
from os.path import dirname, join

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train
import utils
import sys
import random

from vqa_debias_loss_functions import *

torch.set_printoptions(profile="full")

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

def Average(lst):
    return sum(lst) / len(lst)


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
    dataroot = os.path.realpath("data")
    print(dataroot)
    dictionary = Dictionary.load_from_file(join(dataroot, 'dictionary.pkl'))
    cp = not args.nocp

    print("Building train dataset...")

    acount = Counter()
    answer_path = join('data', 'cp-cache', 'train_target.pkl')
    with open(answer_path, 'rb') as f:
        answers = pickle.load(f)
    answers.sort(key=lambda x: x['question_id'])

    # Count the number of occurrence of each answer
    for i in range(len(answers)):
        labels = answers[i]['labels']
        scores = answers[i]['scores']
        if len(scores) == 0:
            continue
        for j in range(len(scores)):
            max_score = max(scores)
            if scores[j] == max_score:
                acount[labels[j]] += 1

    train_index = [i for i in range(len(answers))]

    val_index = []
    count_more_final = Counter()
    question_path = join(
        'data', 'vqacp_v2_train_questions.json')
    with open(question_path) as f:
        questions = json.load(f)
    questions.sort(key=lambda x: x['question_id'])
    index = [i for i in range(len(answers))]
    # random.shuffle(index)
    val_image_ids = []
    limit = 500
    for i in index:
        print(i, len(index))
        labels = answers[i]['labels']
        scores = answers[i]['scores']
        if len(scores) == 0:
            continue
        for j in range(len(scores)):
            max_score = max(scores)
            if scores[j] == max_score:
                label = labels[j]
                true_count = acount[label]
                if label not in count_more_final or count_more_final[label] < limit:
                    count_more_final[label] += 1
                    val_index.append(i)
                    val_image_ids.append(answers[i]['image_id'])
                    train_index.remove(i)
                    break

    # train_index = [i for i in train_index if answers[i]['image_id'] not in val_image_ids]
    print(len(val_index), len(train_index))

    val_questions = [questions[i] for i in val_index]
    # train_questions = [questions[i] for i in train_index]

    val_answers = [answers[i] for i in val_index]

    # train_answers = [answers[i] for i in train_index]

    # with open(join('data', 'vqacp_v2_traindev_questions.json'), "w") as f:
    #     json.dump(train_questions, f, indent=2)

    with open(join('data', 'vqacp_v2_dev_questions.json'), "w") as f:
        json.dump(val_questions, f, indent=2)

    pickle.dump(val_answers, open(join('data', 'cp-cache', 'dev_target.pkl'), 'wb'))
    for i in range(10):
        print(val_questions[i]['question_id'], val_answers[i]['question_id'])
    # pickle.dump(train_answers, open(join('data', 'cp-cache', 'traindev_target.pkl'), 'wb'))
    # Now add a `bias` field to each example
    # for key in question_type_to_prob_array:
    #     print(key, max(question_type_to_prob_array[key].tolist()), sum(question_type_to_prob_array[key].tolist()))
    #     question_type_to_prob_array[key] = question_type_to_prob_array[key].tolist()
    # total = [0] * 2274
    # for key in question_type_to_prob_array:
    #     val = question_type_to_prob_array[key]
    #     total = [total[i] + val[i] for i in range(2274)]
    # print(total)
    # print(min(total), max(total), sum(total)/len(total))
    # with open("probabilities.json", "w") as f:
    #     json.dump(question_type_to_prob_array, f, indent=2)
    # # Build the model using the original constructor
    # constructor = 'build_%s' % args.model
    # model, reconstruction_model = getattr(base_model, constructor)(train_dset, args.num_hid)
    # model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    #
    # # Add the loss_fn based our arguments
    # if args.mode == "bias_product":
    #     model.debias_loss_fn = BiasProduct()
    # elif args.mode == "none":
    #     model.debias_loss_fn = Plain()
    # elif args.mode == "reweight":
    #     model.debias_loss_fn = ReweightByInvBias()
    # elif args.mode == "learned_mixin":
    #     model.debias_loss_fn = LearnedMixin(args.entropy_penalty)
    # else:
    #     raise RuntimeError(args.mode)
    # # constructor = 'build_reconstruction'
    # # reconstruction_model = getattr(base_model, constructor)(args.num_hid).cuda()
    # # Record the bias function we are using
    # utils.create_dir(args.output)
    # with open(args.output + "/debias_objective.json", "w") as f:
    #     js = model.debias_loss_fn.to_json()
    #     json.dump(js, f, indent=2)
    # reconstruction_model = reconstruction_model.cuda()
    # model = model.cuda()
    # # model.apply(weights_init_kn)
    # batch_size = args.batch_size
    #
    #
    # # The original version uses multiple workers, but that just seems slower on my setup
    # train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    # eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)
    #
    # print("Starting training...")
    # # train(model, None, train_loader, eval_loader, args.epochs // 3, args.output, True, seed, 'original')
    # # train(model, reconstruction_model, train_loader, eval_loader, args.epochs // 3, args.output, False, seed,
    # #       'reconstruction')
    # train(model, reconstruction_model, train_loader, eval_loader, args.epochs, args.output, True, seed,
    #       'combination')


if __name__ == '__main__':
    main()