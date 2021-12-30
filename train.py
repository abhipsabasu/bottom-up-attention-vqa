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
from language_model import WordEmbedding
from dataset import Dictionary

torch.autograd.set_detect_anomaly(True)
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def cosine_loss(v, v_recons):
    cos = nn.CosineSimilarity(dim=1)
    obj_dist = cos(v, v_recons)
    # print(v.size(), v_recons.size(), obj_dist.size())
    obj_loss = 1 - obj_dist
    obj_loss = obj_loss.mean()
    return obj_loss


def train(model, train_loader, eval_loader, num_epochs, output, eval_each_epoch):
    ans_embedding = torch.load(os.path.join('data', 'ans_embedding.pth'))
    utils.create_dir(output)
    topv = 1
    top_hint = 9
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
        l1_tot = 0
        t = time.time()
        for i, (v, q, a, b) in tqdm(enumerate(train_loader), ncols=100,
                                    desc="Epoch %d" % (epoch+1), total=len(train_loader)):
            optim.step()
            optim.zero_grad()
            total_step += 1
            v = Variable(v).cuda().requires_grad_()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            b = Variable(b).cuda()
            # hintscore = Variable(hint).cuda()
            bsize = v.size(0)
            if epoch < 0:
                pred, loss, l1 = model(v, None, q, a, b, shuffle=False)
            else:
                pred, loss, l1 = model(v, None, q, a, b, shuffle=True)
                l1_tot += l1.item() * v.size(0)
            # print("1", loss)
            if (loss != loss).any():
                raise ValueError("NaN loss")
            # visual_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), v, create_graph=True)[0]
            # if epoch < 40:
            #     pred, loss = model(v, None, q, a, b, shuffle=False)
            #     # print("1", loss)
            #     if (loss != loss).any():
            #         raise ValueError("NaN loss")
            # else:
            #     pred, pred_shuffle, loss_pos, loss_shuffle = model(v, None, q, a, b, shuffle=True)
            #     loss = loss_pos + 3*loss_shuffle

            # visual_grad_cam = visual_grad.sum(2)
            # hint_sort, hint_ind = hintscore.sort(1, descending=True)
            # v_ind = hint_ind[:, :top_hint]
            # # v_grad = visual_grad_cam.gather(1, v_ind)
            # # v_grad_ind = v_grad.sort(1, descending=True)[1][:, :topv]
            # # v_star = v_ind.gather(1, v_grad_ind)
            # v_ = torch.zeros(v.shape[0], 36).cuda()
            # v_.scatter_(1, v_ind, 1)
            # v_ = v_[:, :, None].expand(bsize, v.shape[1], v.shape[2])
            # v = v * v_
            # v = v[abs(v).sum(dim=2) != 0]
            # v = v.view(bsize, top_hint, -1)
            # v_recons = v_recons.view(bsize, topv, -1)
            # v_recons_j = v_recons_j.view(bsize, topv, -1)
            # # print(v_recons)
            # # print(v)
            # v_max, v_argmax = torch.max(v, 1)
            # v_max = v_max.view(bsize, topv, -1)
            # cos_loss = cosine_loss(v_max, v_recons) + cosine_loss(v_max, v_recons_j)
            # # grad1 = torch.autograd.grad(cos_loss, pred, create_graph=True)[0]
            # # grad2 = torch.autograd.grad(cos_loss, joint, create_graph=True)[0]
            # # print(grad1.sum(), grad2.sum())
            # loss = 3*cos_loss + loss
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score

            # print(l1_tot)
        l1_tot /= len(train_loader.dataset)
        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model_path = os.path.join(output, 'model' + str(epoch) + '.pth')
        torch.save(model.state_dict(), model_path)
        run_eval = True #eval_each_epoch or (epoch == num_epochs - 1)
        for name, p in model.named_parameters():
            if p.grad is not None:
                logger.write(name+' '+str(p.grad.norm()))
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
        logger.write('\ttrain_loss: %.2f, l1 loss: %.2f, score: %.2f' % (total_loss, l1_tot, train_score))

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
        with torch.no_grad():
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            pred, _, _ = model(v, None, q, None, None)
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
