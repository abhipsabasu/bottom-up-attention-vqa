import argparse
import json
from os import listdir
from os.path import join, exists, isdir
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import compute_score_with_logits
import base_model
from dataset import VQAFeatureDataset, Dictionary


def main():
    parser = argparse.ArgumentParser("Save a model's predictions for the VQA-CP test set")
    parser.add_argument("model", help="Directory of the model")
    parser.add_argument("output_file", help="File to write json output to")
    args = parser.parse_args()

    path = args.model

    print("Loading data...")
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, cp=False)
    eval_dset = VQAFeatureDataset('val', dictionary, cp=False)

    eval_loader = DataLoader(eval_dset, 512, shuffle=False, num_workers=0)

    constructor = 'build_%s' % 'baseline0_newatt'
    model,_ = getattr(base_model, constructor)(eval_dset, 1024)

    print("Loading state dict for %s..." % path)

    state_dict = torch.load(join(path, "5193_combination_model.pth"))
    if all(k.startswith("module.") for k in state_dict):
        filtered = {}
        for k in state_dict:
            filtered[k[len("module."):]] = state_dict[k]
        state_dict = filtered

    for k in list(state_dict):
        if k.startswith("debias_loss_fn"):
            del state_dict[k]

    model.load_state_dict(state_dict)

    model.cuda()
    model.eval()
    print("Done")
    score = 0
    predictions = []
    for v, q, a, _, b in tqdm(eval_loader, ncols=100, total=len(eval_loader), desc="eval"):
        with torch.no_grad():
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            factor = model(v, None, q, None, None, True)[1]
            batch_score = compute_score_with_logits(factor, a.cuda()).sum()
            score += batch_score
            prediction = torch.max(factor, 1)[1].data.cpu().numpy()
            for p in prediction:
                predictions.append(train_dset.label2ans[p])
    print(score)
    score = score / len(eval_loader.dataset)
    print(score, len(eval_loader.dataset))
    # out = []
    # for p, e in zip(predictions, eval_dset.entries):
    #     out.append(dict(answer=p, question_id=e["question_id"]))
    # with open(join(path, args.output_file), "w") as f:
    #     json.dump(out, f)


if __name__ == '__main__':
    main()