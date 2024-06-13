import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
def entropy(p):
    p = F.softmax(p, dim=1)
    return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))

def entropy2(p):
    p = F.softmax(p, dim=1)
    return -torch.mean(torch.sum(torch.log(p+1e-5), 1))

def entropy_margin(p, value, margin=0.2, weight=None):
    p = F.softmax(p, dim=1)
    return -torch.mean(hinge(torch.abs(-torch.sum(p * torch.log(p+1e-5), 1)-value), margin))

def hinge(input, margin=0.2):
    return torch.clamp(input, min=margin)

# def vat_loss(C1, feat_t, iter_num, eps, xi, temp):
#     d = torch.rand(feat_t.shape).sub(0.5).to(feat_t.device)
#     logit = C1(feat_t)
#     pred = F.softmax(logit / temp, dim=1).detach().clone()
#     for i in range(iter_num):
#         d = xi * F.normalize(d, p=2)
#         d.requires_grad_()
#         output_grad = C1(F.normalize(feat_t.detach() + d))
#         logp_hat = F.log_softmax(output_grad / temp, dim=1)
#         adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')  # 表示对批次中的所有样本计算KL散度，并返回一个批次的平均值。
#         adv_distance.backward()
#         d = F.normalize(d.grad)
#         C1.zero_grad()
#
#     r_adv = d * eps
#     r_adv = F.normalize(feat_t + r_adv) - feat_t
#     act = feat_t + r_adv.detach().clone()
#     output_grad2 = C1(act)
#     logp_hat2 = F.log_softmax(output_grad2 / temp, dim=1)
#     loss = F.kl_div(logp_hat2, pred, reduction='batchmean')
#     return loss


def vat_loss(C1, feat_t, eps, xi, temp):
    d = torch.rand(feat_t.shape).sub(0.5).to(feat_t.device)
    logit = C1(feat_t)
    pred = F.softmax(logit, dim=1).detach().clone()
    for i in range(1):
        d = xi * F.normalize(d, p=2)
        d = Variable(d.cuda(), requires_grad=True)
        output_grad = C1(F.normalize(feat_t.detach() + d))
        logp_hat = F.log_softmax(output_grad, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')  # 表示对批次中的所有样本计算KL散度，并返回一个批次的平均值。
        adv_distance.backward()
        d = d.clone()
        # C1.zero_grad()
    d = F.normalize(d, p=2)
    d = Variable(d.cuda())
    # print(adv_distance)
    r_adv = d * eps
    # r_adv = F.normalize(feat_t + r_adv) - feat_t
    act = feat_t + r_adv.detach().clone()
    output_grad2 = C1(act)
    logp_hat2 = F.log_softmax(output_grad2, dim=1)
    loss = F.kl_div(logp_hat2, pred, reduction='batchmean')
    return loss

# def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):
#     # find r_adv
#     d = torch.Tensor(ul_x.size()).normal_()
#     for i in range(num_iters):
#         d = xi *_l2_normalize(d)
#         d = Variable(d.cuda(), requires_grad=True)
#         y_hat = model(ul_x + d)
#         delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
#         delta_kl.backward()
#         d = d.clone().cpu()
#         model.zero_grad()
#     d = _l2_normalize(d)
#     d = Variable(d.cuda())
#     r_adv = eps * d
#     # compute lds
#     y_hat = model(ul_x + r_adv.detach())
#     delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
#     return delta_kl


# def contrastive_loss(pos_logit, neg_logit):
#     # pos_len = pos_logit.size(1)
#     logit_ins = torch.cat([pos_logit, neg_logit], dim=1)
#     p = F.softmax(logit_ins, dim=1)
#     p = torch.log(p+torch.tensor(1e-5))
#     sum = torch.sum(p[:, :10], dim=1)
#     loss = - torch.mean(sum, dim=0)
#
#     return loss

def contrastive_loss(pos_logit, neg_logit):
    criterion = nn.CrossEntropyLoss().cuda()
    # pos_len = pos_logit.size(1)
    logit_ins = torch.cat([pos_logit, neg_logit], dim=1)
    labels = torch.zeros(logit_ins.shape[0], dtype=torch.long).cuda()
    loss = criterion(logit_ins, labels)

    return loss