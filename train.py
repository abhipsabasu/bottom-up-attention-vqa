import json
import os
import pickle
import time
from os.path import join
import torch.nn.functional as F
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


# from torch_ema import ExponentialMovingAverage


def binary_acc(y_pred, y_test):
    # print(y_pred.shape, y_test.shape)
    y_pred = y_pred.squeeze(-1)
    y_test = y_test.squeeze(-1)
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag.float() == y_test.float()).sum().float()
    # print(correct_results_sum)
    acc = correct_results_sum  # / y_test.shape[0]
    # print(acc, y_pred_tag.size(), y_test.size())
    # acc = torch.round(acc * 100)

    return acc


def cosine_loss(v, v_recons):
    cos = nn.CosineSimilarity(dim=1)
    obj_dist = cos(v, v_recons)
    # print(v.size(), v_recons.size(), obj_dist.size())
    obj_loss = 1 - obj_dist
    obj_loss = obj_loss.mean()
    return obj_loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, reconstruction_model, train_loader, eval_loader, num_epochs, output, eval_each_epoch, seed, state):
    utils.create_dir(output)
    if state == 'original':
        optim = torch.optim.Adamax(model.parameters(), lr=0.002)
    else:
        optim = torch.optim.Adamax([{'params': model.parameters(), 'lr': 0.002},
                                    {'params': reconstruction_model.parameters(), 'lr': 0.002}])
    # optim = torch.optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08,
    #                          weight_decay=0)
    # scheduler = MultiStepLR(optim, milestones=[5, 7, 9, 11, 13], gamma=0.25)
    # ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    # scheduler.last_epoch = -1
    logger = utils.Logger(os.path.join(output, str(seed) + 'log.txt'))
    logger.write('Without pretraining (1000)')
    all_results = []
    best_score = 0
    total_step = 0
    print('Seed:', seed)
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        l1_tot = 0
        t = time.time()
        print(get_lr(optim))
        mult = 1
        if eval_each_epoch is False:
            mult = 2
        for i, (v, q, a, q_id, b) in tqdm(enumerate(train_loader), ncols=100,
                                          desc="Epoch %d" % (epoch + 1), total=len(train_loader)):
            model.zero_grad()
            if state != 'original':
                reconstruction_model.zero_grad()
            optim.zero_grad()
            total_step += 1
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()

            # hint = Variable(hint).cuda()
            if state == 'original' and eval_each_epoch is False:
                batch_size = v.size(0)
                index = random.sample(range(0, batch_size), batch_size)
                v_neg = v[index]
                v_tot = torch.cat((v, v_neg), dim=0)
                q_tot = torch.cat((q, q), dim=0)
                b_tot = torch.cat((b, b), dim=0)
                a_pos = torch.ones(batch_size, 1).cuda()
                a_neg = torch.zeros(batch_size, 1).cuda()
                a_tot = torch.cat((a_pos, a_neg), dim=0)
                _, pred, _, _, _, loss = model(v_tot, None, q_tot, a_tot, b_tot)
            elif state == 'original' and eval_each_epoch is True:
                _, pred, _, _, _, loss = model(v, None, q, a, b)
            elif state == 'reconstruction':
                pred, v, v_emb, q_emb, _ = model(v, None, q, a, b)
                loss = reconstruction_model(v, pred, q_emb, v_emb, hint)
            else:
                pred_, pred, v, v_emb, q_emb, loss = model(v, None, q, a, b)
                l1 = reconstruction_model(v, pred_, q_emb, v_emb, hint, b)
                loss = loss + l1
                l1_tot += l1.item() * v.size(0)
            if (loss != loss).any():
                raise ValueError("NaN loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()
            # ema.update()
            total_loss += loss.item() * v.size(0)
            if eval_each_epoch is False:
                batch_score = binary_acc(pred, a_tot)  # compute_score_with_logits(pred, a.data).sum()
            else:
                batch_score = compute_score_with_logits(pred, a.data).sum()
            train_score += batch_score

        total_loss /= (mult * len(train_loader.dataset))
        train_score = 100 * train_score / (mult * len(train_loader.dataset))
        l1_tot /= len(train_loader.dataset)
        run_eval = eval_each_epoch  # or (epoch == num_epochs - 1)

        if run_eval:
            model.train(False)
            # with ema.average_parameters():
            if state != 'original':
                reconstruction_model.train(False)
            results = evaluate(model, eval_loader)
            results["epoch"] = epoch + 1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score.item()
            eval_score = results["score"]
            if best_score < eval_score:
                print('High Score! Model saved')
                best_score = eval_score
                model_path = os.path.join(output, str(seed) + '_' + state + '_model.pth')
                torch.save(model.state_dict(), model_path)
            results["best_score"] = best_score
            all_results.append(results)

            with open(join(output, str(seed) + "_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

            model.train(True)
            if state != 'original':
                reconstruction_model.train(True)
            bound = results["upper_bound"]

        logger.write('epoch %d, time: %.2f' % (epoch + 1, time.time() - t))
        if state != 'reconstruction':
            logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
            logger.write('\tl1 loss: %f' % l1_tot)
        else:
            logger.write('\ttrain_loss: %.2f' % total_loss)
        # scheduler.step()
        if run_eval:
            print('Best Score:', best_score * 100, 'seed:', seed)
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
        if state == 'original' and eval_each_epoch is not True:
            if epoch in [4, 14, 49]:
                model_path = os.path.join(output, 'pretraining_' + str(epoch + 1) + '_model.pth')

                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optim.state_dict()}, model_path)


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
            _, pred, _, _, _, _ = model(v, None, q, None, None)
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
