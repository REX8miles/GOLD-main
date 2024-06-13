import numpy as np
import torch
from torch.autograd import Function
from torch import nn
import torch.nn.functional as F

class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]))
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        self.flag = 0
        self.T = T
        self.memory = self.memory.cuda()
    def forward(self, x, y):
        out = torch.mm(x, self.memory.t())/self.T
        return out

    def update_weight(self, features, index):
        if not self.flag:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(0.0)
            weight_pos.add_(torch.mul(features.data, 1.0))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
            self.flag = 1
        else:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(self.momentum)
            weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
        self.memory = F.normalize(self.memory)#.cuda()


    def set_weight(self, features, index):
        self.memory.index_copy_(0, index, features)



class MemoryBank:
    def __init__(self, len):
        # 初始化 feature bank 和 label bank，使用 PyTorch tensor 表示
        self.feature_bank = torch.zeros((len, 2048), dtype=torch.float16).cuda()
        self.label_bank = torch.zeros((len, 1), dtype=torch.int64).cuda()  # 使用 long 类型存储整数标签

    def update_memory(self, indices, features, labels):
        # 使用 index_copy_() 在指定索引位置上更新 feature bank 和 label bank
        self.feature_bank.index_copy_(0, indices, features)
        self.label_bank.index_copy_(0, indices, labels.unsqueeze(1))
