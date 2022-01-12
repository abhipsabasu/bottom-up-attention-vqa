import torch
import torch.nn as nn
from attention import Attention, NewAttention, NewAttentionQ
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import numpy as np
import torch.nn.functional as F
from vqa_debias_loss_functions import Plain
import random
from train import cosine_loss
import block


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return gelu(x)


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def compute_self_loss(logits_neg, a):
    prediction_ans_k, top_ans_ind = torch.topk(F.softmax(a, dim=-1), k=1, dim=-1, sorted=False)
    neg_top_k = torch.gather(F.softmax(logits_neg, dim=-1), 1, top_ans_ind).sum(1)
    qice_loss = neg_top_k.mean()
    return qice_loss


# def instance_bce_with_logits(logits, labels, bias, reduction='mean', focal=False):
#     assert logits.dim() == 2
#     if focal is True:
#         loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
#         loss = (1-bias)**2 * loss
#         if reduction == 'mean':
#             # loss *= labels.size(1)
#             loss = loss.sum() / labels.size(0)
#     else:
#         loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
#         if reduction == 'mean':
#             loss *= labels.size(1)
#     return loss


def instance_bce_with_logits(logits, labels, bias, reduction='mean', focal=False):
    if focal is True:
        # print(logits.sum())
        logits = logits + torch.log(bias**1 + 1e-12)
        # labels = torch.max(labels, 1)[1].data
        # loss = torch.nn.functional.cross_entropy(logits, labels)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        if reduction == 'mean':
            loss *= labels.size(1)
    else:
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        if reduction == 'mean':
            loss *= labels.size(1)
    return loss


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, v_att_1, q_att, q_net, v_net, classifier, num_hid):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_att = q_att
        self.q_net = q_net
        self.v_net = v_net
        self.v_att_1 = v_att_1
        self.classifier = classifier
        self.debias_loss_fn = None
        self.a2v = nn.Sequential(
            nn.Linear(2274, num_hid * 2),
            GeLU(),
            BertLayerNorm(num_hid * 2, eps=1e-12),
            nn.Linear(num_hid * 2, 2048)
        )
        self.q2v = nn.Sequential(
            nn.Linear(1024, num_hid * 2),
            GeLU(),
            BertLayerNorm(num_hid * 2, eps=1e-12),
            nn.Linear(num_hid * 2, 2048)
        )
        # self.j2v = nn.Linear(num_hid, 2*num_hid)
        # self.q2v = nn.Linear(2*num_hid, 2274)
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.normal = nn.BatchNorm1d(num_hid, affine=False)
        self.l1 = torch.nn.L1Loss()

    def forward(self, v, _, q, labels, bias, hintscore, return_weights=False, shuffle=False, focal=False):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        top_hint = 9
        w_emb = self.w_emb(q)
        # w_emb = self.q_att(w_emb, v)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]
        # att = self.v_att(v, q_emb)
        # att_1 = self.v_att_1(v, q_emb)
        # att = att + att_1
        # v_emb = (att * v).sum(1)  # [batch, v_dim]
        #
        batch_size = w_emb.size(0)
        #
        q_repr = self.q_net(q_emb)
        # v_repr = self.v_net(v_emb)
        # joint_repr = q_repr * v_repr
        # joint_repr_normal = self.normal(joint_repr)
        # logits = self.classifier(joint_repr_normal)
        logits,_ = self.compute_predict(q_repr, q_emb, v)
        l1 = 0
        if labels is not None:
            if return_weights:
                return self.debias_loss_fn(joint_repr, logits, bias, labels, True)
            # if shuffle is True:
            #     loss = compute_self_loss(logits, labels)
            # else:
            loss = instance_bce_with_logits(logits, labels, bias, focal=focal)

            # if shuffle:
            #     # hint_sort, hint_ind = hintscore.sort(1, descending=True)
            #     # v_ind = hint_ind[:, :top_hint]
            #     # v_ = torch.zeros(v.shape[0], 36).cuda()
            #     # v_.scatter_(1, v_ind, 1)
            #     # v_ = v_[:, :, None].expand(batch_size, v.shape[1], v.shape[2])
            #     # v_ = v * v_
            #     # # print(v_.size(), v_ind.size())
            #     # v_ = v_[abs(v_).sum(dim=2) != 0]
            #     # v_ = v_.view(batch_size, top_hint, -1)
            #     v_max = torch.mean(v, 1)
            #     v_recons = self.a2v(logits) + self.q2v(q_emb)
            #     v_recons = v_recons.unsqueeze(1)
            #     v_emb_r = (att * v_recons).sum(1)
            #     v_repr_r = self.v_net(v_emb_r)
            #     joint_repr_r = q_repr * v_repr_r
            #     joint_repr_normal_r = self.normal(joint_repr_r)
            #     logits_r = self.classifier(joint_repr_normal_r)
            #     l1 = 3 * self.l1(v_max, v_recons)
            #     cycle = self.debias_loss_fn(joint_repr_r, logits_r, bias, labels)
            #     loss = (loss + l1 + cycle)  # + instance_bce_with_logits(logits1, labels)
        else:
            loss = None
        if shuffle:
            # construct an irrelevant Q-I pair for each instance
            # loss = instance_bce_with_logits(logits, labels, bias, focal=focal)
            # v_max = torch.mean(v, 1)
            # v_recons = self.a2v(logits) + self.q2v(q_emb)
            # l1 = 3 * self.l1(v_max, v_recons)
            # loss = loss + l1
            index = random.sample(range(0, batch_size), batch_size)
            v_neg = v[index]
            logits_neg,_ = self.compute_predict(q_repr, q_emb, v_neg)
            # att_neg = self.v_att(v_neg, q_emb)
            # att_1_neg = self.v_att_1(v_neg, q_emb)
            # att_neg = att_neg + att_1_neg
            # v_emb_neg = (att_neg * v_neg).sum(1)  # [batch, v_dim]
            # # q_repr = self.q_net(q_emb)
            # v_repr_neg = self.v_net(v_emb_neg)
            # joint_repr_neg = q_repr * v_repr_neg
            # joint_repr_normal_neg = self.normal(joint_repr_neg)
            # logits_neg = self.classifier(joint_repr_normal_neg)
            loss_neg = compute_self_loss(logits_neg, labels)
            return logits, logits_neg, loss, loss_neg, l1
        return logits, loss, l1

    def compute_predict(self, q_repr, q_emb, v):

        att_1 = self.v_att(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.v_att_1(v, q_emb)  # [batch, 1, v_dim]
        att_gv = att_1 + att_2

        gv_embs = (att_gv * v)  # [batch, v_dim]
        gv_emb = gv_embs.sum(1)
        gv_repr = self.v_net(gv_emb)

        joint_repr = q_repr * gv_repr

        # joint_repr_normal = self.normal(joint_repr)
        logits = self.classifier(joint_repr)

        return logits, att_gv

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
    q_att = NewAttentionQ(300, dataset.v_dim, 300)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    v_att_1 = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, v_att_1, q_att, q_net, v_net, classifier, num_hid)
