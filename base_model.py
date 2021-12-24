import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import numpy as np
import torch.nn.functional as F
from vqa_debias_loss_functions import Plain
import random


def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)

    qice_loss = neg_top_k.mean()
    return qice_loss

def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, num_hid):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.debias_loss_fn = None
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.normal = nn.BatchNorm1d(num_hid, affine=False)

    def forward(self, v, _, q, labels, bias, return_weights=False, shuffle=False):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        batch_size = v_emb.size(0)
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        joint_repr_normal = self.normal(joint_repr)
        logits = self.classifier(joint_repr_normal)
        if labels is not None:
          if return_weights:
            return self.debias_loss_fn(joint_repr, logits, bias, labels, True)
          # if shuffle is True:
          #     loss = compute_self_loss(logits, labels)
          # else:
          loss = instance_bce_with_logits(logits, labels)
        else:
          loss = None
        if shuffle:
            # construct an irrelevant Q-I pair for each instance
            index = random.sample(range(0, batch_size), batch_size)
            v_neg = v[index]
            att = self.v_att(v_neg, q_emb)
            v_emb = (att * v_neg).sum(1)  # [batch, v_dim]
            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            joint_repr_normal = self.normal(joint_repr)
            logits_neg = self.classifier(joint_repr_normal)
            loss_neg = compute_self_loss(logits_neg, labels)
            return logits, logits_neg, loss, loss_neg

        return logits, loss


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, num_hid)
