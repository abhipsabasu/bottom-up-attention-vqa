import json
import os
import pickle
import time
from os.path import join
import random
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

torch.autograd.set_detect_anomaly(True)
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output, eval_each_epoch):
    utils.create_dir(output)
    optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=0)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    all_results = []
    scheduler = MultiStepLR(optim, milestones=[10, 15, 20, 25, 30, 35], gamma=0.5)
    scheduler.last_epoch = 0
    total_step = 0
    best_score = 0
    for epoch in range(40):
        total_loss = 0
        train_score = 0

        t = time.time()
        for i, (v, q, a, b) in tqdm(enumerate(train_loader), ncols=100,
                                    desc="Epoch %d" % (epoch+1), total=len(train_loader)):
            total_step += 1
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()
            bsize = v.size(0)
            shuffle_index = random.sample(range(0, bsize), bsize)
            # print(shuffle_index)
            # v_shuffle = v[shuffle_index]
            if epoch < 12:
                pred, loss = model(v, None, q, a, b, shuffle=False)
                # print("1", loss)
                if (loss != loss).any():
                    raise ValueError("NaN loss")
            else:
                pred, pred_shuffle, loss_pos, loss_shuffle = model(v, None, q, a, b, shuffle=True)
                loss = loss_pos + 3*loss_shuffle
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score
            # if epoch >= 5:
            #     pred_shuffle, loss_shuffle = model(v_shuffle, None, q, a, b, shuffle=True)
            #     loss_shuffle = loss_shuffle.sum()/bsize
            #     # print("2", loss_shuffle)
            #     if (loss != loss).any():
            #         raise ValueError("NaN loss")
            #     loss_shuffle = 3*loss_shuffle
            #     loss_shuffle.backward()
            #     nn.utils.clip_grad_norm(model.parameters(), 0.25)
            #     optim.step()
            #     optim.zero_grad()

            # batch_score = compute_score_with_logits(pred_shuffle, a.data).sum()
            # total_loss += loss_shuffle.item() * v.size(0)
            # train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        run_eval = True #eval_each_epoch or (epoch == num_epochs - 1)

        if run_eval:
            model.train(False)
            results = evaluate(model, eval_loader)
            results["epoch"] = epoch+1
            results["step"] = total_step
            results["train_loss"] = total_loss
            results["train_score"] = train_score.item()
            all_results.append(results)
            print(all_results)
            with open(join(output, "results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

            model.train(True)

            eval_score = results["score"]
            bound = results["upper_bound"]
        scheduler.step()
        logger.write('epoch %d, time: %.2f' % (epoch+1, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))

        if run_eval:
            if eval_score > best_score:
                best_score = eval_score
                model_path = os.path.join(output, 'model.pth')
                torch.save(model.state_dict(), model_path)
                print("High score! model saved.")
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

    # model_path = os.path.join(output, 'model.pth')
    # torch.save(model.state_dict(), model_path)


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0

    all_logits = []
    all_bias = []
    for v, q, a, b in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred, _ = model(v, None, q, None, None)
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
