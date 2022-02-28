import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm
from train import cosine_loss
import torch.nn.functional as F
import os
from vqa_debias_loss_functions import *


# def instance_bce_with_logits(logits, labels, bias=None, reduction='mean'):
#     # assert logits.dim() == 2
#     #
#     # loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
#     # if reduction == 'mean':
#     #     loss *= labels.size(1)
#     loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
#     loss = (1 - bias) ** 2 * loss
#     if reduction == 'mean':
#         # loss *= labels.size(1)
#         loss = loss.sum() / labels.size(0)
#     return loss

def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


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


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, num_hid, a_token):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.debias_loss_fn = LearnedMixin(0.36)
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.normal = nn.BatchNorm1d(num_hid, affine=False)

    def forward(self, v, _, q, labels, bias, return_weights=False):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        # [batch, v_dim]
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        joint_repr_normal = self.normal(joint_repr)
        logits = self.classifier(joint_repr_normal)
        if labels is not None:
            if return_weights:
                return self.debias_loss_fn(joint_repr, logits, bias, labels, True)
            # labels = labels.unsqueeze(-1)
            labels = labels.long()
            loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
            loss *= labels.size(1)
        else:
          loss = None
        return joint_repr_normal, logits, joint_repr, None, att, loss



class ClassifierModel(nn.Module):
    def __init__(self, basemodel, classifier):
        super(ClassifierModel, self).__init__()
        self.basemodel = basemodel
        self.classifier = classifier

    def forward(self, v, _, q, labels, bias, return_weights=False):
        hidden, _, _, _, _, _ = self.basemodel(v, _, q, None, bias, return_weights)
        logits = self.classifier(hidden)
        if labels is not None:
            if return_weights:
                return self.debias_loss_fn(joint_repr, logits, bias, labels, True)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss *= labels.size(1) #self.debias_loss_fn(joint_repr, logits, bias, labels)
            # loss *= labels.size(1)
        else:
          loss = None
        return None, logits, None, None, None, loss


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
    classifier1 = SimpleClassifier(num_hid, num_hid * 2, 1, 0.5)
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

    basemodel = BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier1, num_hid, None)
    model_path = os.path.join('saved_models/exp0', 'pretraining_5_model.pth')
    model_data = torch.load(model_path)
    basemodel.load_state_dict(model_data['model_state_dict'])
    basemodel.w_emb.init_embedding('data/glove6b_init_300d.npy')
    for p in basemodel.classifier.parameters():
        p.requires_grad = False

    classifierModel = ClassifierModel(basemodel, classifier)
    return classifierModel, None


def build_pretrain(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier1 = SimpleClassifier(num_hid, num_hid * 2, 1, 0.5)

    basemodel = BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier1, num_hid, None)
    return basemodel, None

def build_original(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    basemodel = BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, num_hid, None)
    return basemodel, None
