import torch, math
from bisect import bisect_right

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler): # 用于调整学习率
    def __init__(
        self,
        optimizer,  # 优化器
        milestones, # 调整学习率的epoch数
        gamma=0.1,  # γ值？没懂
        warmup_factor=1.0 / 3,  # warmup因子：1/3
        warmup_epochs=5,    # warmup epoch？5
        warmup_method="linear", # warmup方法：线性
        last_epoch=-1,  # 最后epoch：一直到最后
    ):
        if not list(milestones) == sorted(milestones):  # 检查milestone是否为单调递增的整数列表
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"): # 检查warmup方法是否为constant或linear之一
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]