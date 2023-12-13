import torch
import torch.nn as nn
from torch.nn import functional as F


def NBOD(inputs, factor):
    classifier_num = len(inputs)
    if classifier_num == 1:
        return 0
    logits_softmax = []
    logits_logsoftmax = []
    for i in range(classifier_num):
        logits_softmax.append(F.softmax(inputs[i], dim=1))
        logits_logsoftmax.append(torch.log(logits_softmax[i] + 1e-9))

    loss_mutual = 0
    for i in range(classifier_num):
        for j in range(classifier_num):  # 三个专家两两互相蒸馏，一共蒸馏6次
            if i == j:
                continue
            loss_mutual += factor * F.kl_div(logits_logsoftmax[i], logits_softmax[j], reduction='batchmean')  # 第0个入参是待度量分布Q；第1个入参是目标分布P
    loss_mutual /= (classifier_num - 1)
    return loss_mutual


class NIL_NBOD(nn.Module):
    def __init__(self, para_dict=None):
        super(NIL_NBOD, self).__init__()
        self.para_dict = para_dict
        self.num_class_list = self.para_dict['num_class_list']
        self.device = self.para_dict['device']
        self.bsce_weight = torch.FloatTensor(self.num_class_list).to(self.device)  # 平衡softmax交叉熵的权重

        self.multi_classifier_diversity_factor = self.para_dict['cfg'].LOSS.MULTI_CLASIIFIER_LOSS.DIVERSITY_FACTOR  # 论文中的β
        self.multi_classifier_diversity_factor_hcm = self.para_dict['cfg'].LOSS.MULTI_CLASIIFIER_LOSS.DIVERSITY_FACTOR_HCM
        self.hcm_N = self.para_dict['cfg'].LOSS.HCM_N  # 困难类别选前30个
        self.hcm_ratio = self.para_dict['cfg'].LOSS.HCM_RATIO
        self.ce_ratio = self.para_dict['cfg'].LOSS.CE_RATIO

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        classifier_num = len(inputs)  # 分类器个数
        loss_HCM = 0  # 困难类别挖掘损失
        loss = 0
        los_ce = 0

        inputs_HCM_balance = []
        inputs_balance = []
        class_select = inputs[0].scatter(1, targets[0].unsqueeze(1), 999999)  # 把真实标签位置的值更新成999999
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_N]  # 先对logits从大到小排列，那么真实标签必排在第0位，再截取前30大的作为困难类别
        mask = torch.zeros_like(inputs[0]).scatter(1, class_select_include_target, 1)  # 掩码，前30困难类索引所在位置更新成1，其他更改为0
        for i in range(classifier_num):
            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()  # BSCE损失函数中乘对应类别的样本数(对数乘法变加法)
            inputs_balance.append(logits)
            inputs_HCM_balance.append(logits * mask)  # 乘困难类掩码，留下前30的困难logits

            los_ce += F.cross_entropy(logits, targets[0])  # 计算NIL模块全部类别的损失
            loss_HCM += F.cross_entropy(inputs_HCM_balance[i], targets[0])  # 计算NIL模块困难类别的损失

        loss += NBOD(inputs_balance, factor=self.multi_classifier_diversity_factor)  # 计算NBOD模块全部类别的损失
        loss += NBOD(inputs_HCM_balance, factor=self.multi_classifier_diversity_factor_hcm)  # 计算NBOD模块困难类别的损失
        loss += los_ce * self.ce_ratio + loss_HCM * self.hcm_ratio
        return loss

    def update(self, epoch):
        """
        Args:
           code can be added for progressive loss.
        """
        pass


if __name__ == '__main__':
    pass
