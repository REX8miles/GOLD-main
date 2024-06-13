import torch

def compute_updated_prototypes(features, labels, prototypes_memory, class_sums, class_counts):
    """
    计算更新后的原型

    参数：
        features：包含特征向量的张量，形状为 (batch_size, feature_dim)，
                  其中 batch_size 是批次大小，feature_dim 是特征维度。
        labels：包含样本类别标签的张量，形状为 (batch_size,)，
                其中 batch_size 是批次大小，取值范围为 [0, num_classes-1]。
        prototypes_memory：之前计算好的原型张量，形状为 (num_classes, feature_dim)，
                           其中 num_classes 是类别数量，feature_dim 是特征维度。
        class_sums：之前计算好的每个类别的特征总和张量，形状为 (num_classes, feature_dim)。
        class_counts：之前计算好的每个类别的样本数量张量，形状为 (num_classes,)。

    返回：
        updated_prototypes：更新后的原型张量，形状为 (num_classes, feature_dim)。
    """
    # 将特征张量和原型张量移动到CUDA上（如果还没有在CUDA上）
    features = features.float().cuda()
    prototypes_memory = prototypes_memory.cuda()

    # 使用标签对应的索引更新class_sums和class_counts
    class_sums.index_add_(0, labels, features)
    class_counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32).cuda())

    # 根据更新后的class_sums和class_counts计算更新后的原型
    updated_prototypes = class_sums / class_counts.unsqueeze(1)
    updated_prototypes = torch.where(torch.isnan(updated_prototypes), torch.full_like(updated_prototypes, 0), updated_prototypes)

    return updated_prototypes
