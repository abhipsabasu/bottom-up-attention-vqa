import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import numpy as np
from torch.autograd import Variable
from train import cosine_loss


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
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
        self.classifier = classifier
        self.debias_loss_fn = None
        # self.bias_scale = torch.nn.Parameter(torch.from_numpy(np.ones((1, ), dtype=np.float32)*1.2))
        self.bias_lin = torch.nn.Linear(1024, 1)
        self.l1 = torch.nn.L1Loss()
        # self.a_token = Variable(a_token).cuda()

    def forward(self, v, _, q, labels, bias, hint, return_weights=False):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]
        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        # v_mean = torch.mean(v, 1)
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        batch_size = v.size(0)
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)

        if labels is not None:
            if return_weights:
                return self.debias_loss_fn(joint_repr, logits, bias, labels, True)
            loss = self.debias_loss_fn(joint_repr, logits, bias, labels)
            # hint_sig = torch.sigmoid(hint)
            # prediction_ans_k, top_ans_ind = torch.topk(torch.softmax(logits, dim=-1), k=1, dim=-1, sorted=False)
            # ans = self.a_token[[top_ans_ind.squeeze().tolist()]]
            # ans = self.w_emb(ans.long()).mean(1)
            v_recons = self.a2v(logits) + self.q2v(q_emb)
            hint_sig = torch.sigmoid(hint)
            hint_expand = hint_sig[:, :, None].expand(batch_size, v.shape[1], v.shape[2])
            v_weighted = (v * hint_expand).mean(1)
            l1 = 3*self.l1(v_recons, v_weighted) #instance_bce_with_logits(hint_pred, hint_sig) / 2
            loss = loss + l1

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