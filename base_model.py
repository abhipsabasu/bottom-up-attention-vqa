import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import numpy as np
from torch.autograd import Variable
from train import cosine_loss


def instance_bce_with_logits(logits, labels, bias=None, reduction='mean'):
    # assert logits.dim() == 2
    #
    # loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
    # if reduction == 'mean':
    #     loss *= labels.size(1)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    loss = (1 - bias) ** 2 * loss
    if reduction == 'mean':
        # loss *= labels.size(1)
        loss = loss.sum() / labels.size(0)
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
        self.top_hint = 10
        self.a_token = a_token.cuda()
        self.a2v = nn.Sequential(
            nn.Linear(2274, num_hid * 2),
            GeLU(),
            BertLayerNorm(num_hid * 2, eps=1e-12),
            nn.Linear(num_hid * 2, 2048 * self.top_hint)
        )
        self.q2v = nn.Sequential(
            nn.Linear(1024, num_hid * 2),
            GeLU(),
            BertLayerNorm(num_hid * 2, eps=1e-12),
            nn.Linear(num_hid * 2, 2048 * self.top_hint)
        )
        # self.v2v = nn.Sequential(
        #     nn.Linear(2048, num_hid * 2),
        #     GeLU(),
        #     BertLayerNorm(num_hid * 2, eps=1e-12),
        #     nn.Linear(num_hid * 2, 2048 * self.top_hint)
        # )
        self.v_emb2v = nn.Sequential(
            nn.Linear(2048, num_hid * 2),
            GeLU(),
            BertLayerNorm(num_hid * 2, eps=1e-12),
            nn.Linear(num_hid * 2, 2048 * self.top_hint)
        )
        # self.W_att = nn.Sequential(
        #     nn.Linear(2048 * self.top_hint, num_hid),
        #     GeLU(),
        #     BertLayerNorm(num_hid, eps=1e-12),
        #     nn.Linear(num_hid, 2048 * self.top_hint)
        # )
        # self.w_att = nn.Linear(2048 * self.top_hint, 1)
        self.classifier = classifier
        self.debias_loss_fn = None
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.l1 = torch.nn.L1Loss()
        self.normal = nn.BatchNorm1d(num_hid, affine=False)
        # self.recons = nn.BatchNorm1d(2048 * self.top_hint, affine=False)
        # self.cos = nn.CosineSimilarity(dim=-1)
        # self.a_token = Variable(a_token).cuda()

    def forward(self, v, _, q, labels, bias, hint, return_weights=False, cycle=False):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        # top_hint = 36
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        # v_mean = torch.mean(v, 1)
        batch_size = v.size(0)
        arange = torch.arange(batch_size).unsqueeze(1)
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        joint_repr_normal = self.normal(joint_repr)
        logits = self.classifier(joint_repr_normal)
        if labels is not None:
            if return_weights:
                return self.debias_loss_fn(joint_repr, logits, bias, labels, True)
            loss = self.debias_loss_fn(joint_repr, logits, bias, labels)
            hint_sort, hint_ind = hint.sort(1, descending=True)
            v_ind = hint_ind[:, :self.top_hint]
            v_ = v[arange, v_ind]

            # a2v = self.a2v(logits).unsqueeze(-1)
            # q2v = self.q2v(q_emb).unsqueeze(-1)
            # cat = torch.cat((a2v, q2v), -1).permute(0, 2, 1)
            # res = self.W_att(cat)
            # res = torch.tanh(res)
            # res = self.w_att(res)
            # # print(res)
            # res = torch.softmax(res, 1)
            # # print(res.shape)
            # res = res.permute(0, 2, 1)
            # v_recons = torch.bmm(res, cat)

            v_recons = (self.a2v(logits) + self.q2v(q_emb)) / 2
            v_recons = v_recons.view(batch_size, self.top_hint, -1)

            v_emb_rec = self.v_emb2v(v_emb)
            v_emb_rec = v_emb_rec.view(batch_size, self.top_hint, -1)

            l1 = 3 * self.l1(v_recons, v_)  # instance_bce_with_logits(v_recons, v_weighted) / 2
            l1_emb = 3 * self.l1(v_emb_rec, v_)
            loss = loss + l1 + l1_emb

            if cycle is True:
                att = self.v_att(v_emb_rec, q_emb)
                v_emb_cyc = (att * v_emb_rec).sum(1)
                v_repr_cyc = self.v_net(v_emb_cyc)
                joint_repr_cyc = q_repr * v_repr_cyc
                joint_repr_normal_cyc = self.normal(joint_repr_cyc)
                logits_cyc = self.classifier(joint_repr_normal_cyc)
                cyc_loss = instance_bce_with_logits(logits_cyc, labels, bias=bias)
                loss = loss + cyc_loss  # + loss_nce) / 3

        else:
          loss = None
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
    a_token = dataset.ans_tokens
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, num_hid, a_token)