import torch
import numpy as np
from numpy.linalg import norm, pinv
# from torch.linalg import pinv
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.covariance import EmpiricalCovariance
from os.path import basename, splitext
from scipy.special import logsumexp



def get_space(feat_s, C1):
    # 加载特征数据
    feature_id_source = feat_s

    # 计算logits（未经softmax激活的模型输出）
    # feature_id_source = torch.tensor(feature_id_source).cuda()
    logit_id_source = C1(feature_id_source)  # (15 x 15))


    u = feature_id_source.mean(dim=0)
    feature_id_source = feature_id_source - u
    DIM = 1000 if feature_id_source.shape[-1] >= 2048 else 512

    # 计算主空间
    feature_id_source = feature_id_source.cpu().detach().numpy()


    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_source)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    NS = torch.tensor(NS).cuda()
    NS = F.normalize(NS, dim=0)

    # 计算alpha
    # logit_id_source = logit_id_source.cpu().detach().numpy()  # logit_id_source.shape = (36*65)
    # vlogit_id_source = norm(np.matmul(feature_id_source, NS), axis=-1)  # 计算在主空间上的投影
    # alpha = logit_id_source.max(axis=-1).mean() / vlogit_id_source.mean()  # 计算alpha值
    return NS, u



def get_ood_score(feat_t, C1, NS, u):
    feature_ood_target = feat_t     # 36*2048
    logit_ood_target = C1(feature_ood_target)   # 36*15
    feature_ood_target = feature_ood_target.cuda()
    # NS = torch.tensor(NS).cuda()    # 2048*1048

    vlogit_t = torch.norm(torch.matmul(feature_ood_target - u, NS), dim=-1)
    # print(norm(np.matmul(feature_ood_target, NS), axis=-1))
    energy_t = torch.logsumexp(logit_ood_target / 0.05, dim=-1)
    score_ood = vlogit_t

    # # 归一化
    # max_score = torch.max(score_ood)
    # min_score = torch.min(score_ood)
    # score = (score_ood - min_score) / (max_score - min_score)
    # 标准化
    # score_std = torch.std(score_ood, dim=-1)
    # score_mean = torch.mean(score_ood)
    # score = (score_ood - score_mean) / score_std
    return score_ood




# def get_space(feat_s, C1):
#     w = C1.fc.weight.data
#     b = C1.fc.bias
#     # 加载特征数据并移动到GPU上
#     feature_id_source = feat_s.cuda()
#
#     # 计算logits（未经softmax激活的模型输出）
#     logit_id_source = C1(feature_id_source)
#
#     if b is not None:
#         u = -torch.matmul(pinv(w).cuda(), b.cuda())
#     else:
#         u = None
#
#     DIM = 1000
#
#     # 计算主空间
#     if u:
#         feature_id_source = feature_id_source - u
#
#     ec = EmpiricalCovariance(assume_centered=True)
#     ec.fit(feature_id_source.cpu().detach().numpy())
#     eig_vals, eigen_vectors = torch.linalg.eigh(torch.tensor(ec.covariance_, device='cuda:0'))
#     _, indices = torch.sort(eig_vals * -1)
#     indices = indices.squeeze()
#     NS = eigen_vectors[:, -DIM:][:, indices]
#
#     # 计算alpha
#     vlogit_id_source = torch.norm(torch.matmul(feature_id_source, NS), dim=-1)
#     alpha = torch.tensor(logit_id_source.max(axis=-1).mean(), device='cuda:0') / vlogit_id_source.mean()
#
#     return NS, alpha
#
#
#
# def get_ood_score(feat_t, C1, NS, alpha):
#     feature_ood_target = feat_t.cuda()
#     logit_ood_target = C1(feature_ood_target)
#     NS=NS.cuda()
#     vlogit_t = torch.norm(torch.matmul(feature_ood_target, NS), dim=-1) * alpha
#
#     energy_t = torch.logsumexp(logit_ood_target, dim=-1)
#     score_ood = -vlogit_t + energy_t
#
#     # 归一化（使用PyTorch张量的方法）
#     max_score = torch.max(score_ood)
#     min_score = torch.min(score_ood)
#     score = (score_ood - min_score) / (max_score - min_score)
#
#     return score