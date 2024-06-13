import torch
import numpy as np
from numpy.linalg import norm
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance



def get_space(feat_id, C1):

    logit_id = C1(feat_id)  # (15 x 15))

    u = feat_id.mean(dim=0)

    feat_id = feat_id - u
    DIM = 1000 if feat_id.shape[-1] >= 2048 else 512

    # 计算主空间
    feat_id = feat_id.cpu().detach().numpy()
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feat_id)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    NS = torch.tensor(NS).cuda()
    NS = F.normalize(NS, dim=0)
    feat_id = torch.tensor(feat_id).cuda()

    # 计算alpha
    vlogit_id = torch.norm(torch.matmul(feat_id, NS), dim=-1)

    vlogit_id = vlogit_id.cpu().detach().numpy()
    logit_id = logit_id.cpu().detach().numpy()

    alpha = logit_id.max(axis=-1).mean() / vlogit_id.mean()
    return NS, alpha, u


def get_ood_score(feat_ood, C1, NS, alpha, u):

    logit_ood = C1(feat_ood)

    vlogit_ood = torch.norm(torch.matmul(feat_ood - u, NS), dim=-1) * alpha
    energy_t = torch.logsumexp(logit_ood, dim=-1)
    score_ood = -vlogit_ood + energy_t

    return score_ood