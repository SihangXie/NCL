import logging
import time
import os

import torch
from utils.lr_scheduler import WarmupMultiStepLR
from net import multi_Network, multi_Network_MOCO


# 训练日志输出保存
def create_logger(cfg, rank=0):
    dataset = cfg.DATASET.DATASET  # 获取数据集名称
    net_type = cfg.BACKBONE.TYPE  # 获取配置文件中backbone的类型
    module_type = cfg.MODULE.TYPE  # 获取配置文件中module的类型，没懂GAP是什么意思
    log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")  # 定义日志储存目录
    if not os.path.exists(log_dir) and rank == 0:  # 如果日志目录不存在就创建该目录
        os.makedirs(log_dir)  # 根据目录路径创建文件夹
    time_str = time.strftime("%Y-%m-%d-%H-%M")  # 返回字符串类型的当前日期时间
    log_name = "{}_{}_{}_{}.log".format(dataset, net_type, module_type, time_str)  # 原来上面获取的信息都是用来创建日志文件的名称
    log_file = os.path.join(log_dir, log_name)  # 把日志目录路径与日志文件名拼接在一起就成了日志文件的绝对路径
    # set up logger 建立日志记录器
    print("=> creating log {}".format(log_file))  # 在控制台打印日志文件的绝对路径
    head = "%(asctime)-15s %(message)s"  # ？没看懂
    logging.basicConfig(filename=str(log_file), format=head)  # 设置日志文件名和记录头等基础设置
    logger = logging.getLogger()  # 获取日志记录器的实例对象
    logger.setLevel(logging.INFO)
    if rank > 0:
        return logger, log_file
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file


def get_optimizer(cfg, model):  # 获取优化器
    base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR  # 从配置文件参数中读取基础学习率：0.1
    params = []  # 创建一个空的参数列表{list:0}，存放模型每层的参数，最后一共是3+15*6*3+6*2=285层参数

    for name, p in model.named_parameters():  # 对model中的每一层的名称name和参数p
        if p.requires_grad:  # 以第一层卷积层为例，3通道，每个通道16个3×3的卷积核，一共48个3×3的卷积核，可学习的参数为48×3×3=432个
            params.append({"params": p})  # 如果需要计算梯度，就把这层的参数加入到参数列表params
        else:
            print("not add to optimizer: {}".format(name))  # 自监督学习的滑动平均模型的主干网络和分类器参数都不参与优化

    if cfg.TRAIN.OPTIMIZER.TYPE == "SGD":  # 如果优化器类型是随机梯度下降SGD就进
        optimizer = torch.optim.SGD(  # 实例化SGD 优化器
            params,  # 285层的模型参数列表
            lr=base_lr,  # 基础学习率：0.1
            momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,  # 动量：0.9
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,  # 权重衰减：2e-4
            nesterov=True,
        )
    elif cfg.TRAIN.OPTIMIZER.TYPE == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError
    return optimizer


def get_scheduler(cfg, optimizer):  # 获取调度器
    if cfg.TRAIN.LR_SCHEDULER.TYPE == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "cosine":
        if cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END, eta_min=1e-4
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.MAX_EPOCH, eta_min=1e-4
            )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "warmup":  # 训练学习率调度类型为warmup进
        scheduler = WarmupMultiStepLR(  # 用于调整学习率
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
            warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARM_EPOCH,  # warmup_epoch为5
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))

    return scheduler


def get_multi_model_final(cfg, num_classes, num_class_list, device, logger):
    if cfg.NETWORK.MOCO:
        model = multi_Network_MOCO(cfg, mode="train", num_classes=num_classes, use_dropout=cfg.DROPOUT)
    else:  # CIFAR100-LT使用的模型网络
        model = multi_Network(cfg, mode="train", num_classes=num_classes, use_dropout=cfg.DROPOUT)

    if cfg.BACKBONE.FREEZE == True:  # 固定backbone参数进
        model.freeze_multi_backbone()
        logger.info("Backbone has been freezed")

    return model


def get_category_list(annotations, num_classes, cfg):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list
