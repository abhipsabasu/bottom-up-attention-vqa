import json
import os
import pickle
import time
from os.path import join

import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR


def cosine_loss(v, v_recons):
    cos = nn.CosineSimilarity(dim=1)
    obj_dist = cos(v, v_recons)
    # print(v.size(), v_recons.size(), obj_dist.size())
    obj_loss = 1 - obj_dist
    obj_loss = obj_loss.mean()
    return obj_loss

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, train_loader, eval_loader, num_epochs, output, eval_each_epoch, seed):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    # optim = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08,
    #                          weight_decay=0)
    scheduler = MultiStepLR(optim, milestones=[15, 20, 25, 30, 35], gamma=0.5)
    scheduler.last_epoch = 0
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    all_results = []
    best_score = 0
    total_step = 0
    print('Seed:', seed)
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0

        t = time.time()
        print(get_lr(optim))
        for i, (v, q, a, q_id, hint, b) in tqdm(enumerate(train_loader), ncols=100,
                                    desc="Epoch %d" % (epoch+1), total=len(train_loader)):
            total_step += 1
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()
            hint = Variable(hint).cuda()
            # if epoch >= 8:
            pred, loss = model(v, None, q, a, b, hint, cycle=False)
            # else:
            #     pred, loss = model(v, None, q, a, b, hint, cycle=False)
            if (loss != loss).any():
              raise ValueError("NaN loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        run_eval = True  # eval_each_epoch or (epoch == num_epochs - 1)

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader)
            results["epoch"] = epoch+1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score.item()
            eval_score = results["score"]
            if best_score < eval_score:
                print('High Score! Model saved')
                best_score = eval_score
                model_path = os.path.join(output, str(seed) + '_model.pth')
                torch.save(model.state_dict(), model_path)
            results["best_score"] = best_score
            all_results.append(results)

            with open(join(output, str(seed) + "_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

            model.train(True)

            bound = results["upper_bound"]

        logger.write('epoch %d, time: %.2f' % (epoch+1, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        scheduler.step()
        if run_eval:

            print('Best Score:', best_score*100, 'seed:', seed)
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0

    all_logits = []
    all_bias = []
    for v, q, a, _, b in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        with torch.no_grad():
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            pred, _ = model(v, None, q, None, None, None, cycle=False)
            all_logits.append(pred.data.cpu().numpy())

            batch_score = compute_score_with_logits(pred, a.cuda()).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)
            all_bias.append(b)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    results = dict(
        score=score.item(),
        upper_bound=upper_bound.item(),
    )
    return results