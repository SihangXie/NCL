import time

from torch.nn import init

import _init_paths

from loss import *
from dataset import *
from config import cfg, update_config
from utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_multi_model_final,
    get_category_list,
)

from core.function import multi_networks_train_model, multi_network_valid_model_final
from core.combiner import Combiner

import torch
import os
import shutil
from torch.utils.data import DataLoader
import argparse
import warnings
import click
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import ast
from datetime import datetime

import random
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="code for NCL")  # 实例化解析器对象，用于从命令行上解析用户输入的参数
    parser.add_argument(  # 解析器对象的方法：新增参数
        "--cfg",  # 可选参数名称，带-前缀的都是可选参数
        help="decide which cfg to use",  # 帮助说明：通常是描述这个参数的作用。此参数是用于指定训练配置文件
        required=False,  # True为不可省略，其实就相当于变成位置参数；False为可省略
        # default="/home/lijun/papers/NCL/config/CIFAR/CIFAR100/cifar100_im100_NCL_with_contrastive.yaml",  # 作者原来的配置文件加载路径
        default="/home/og/XieSH/xsh/NCL/config/CIFAR/CIFAR100/cifar100_im100_NCL_with_contrastive.yaml",  # 参数的默认值，改成自己服务器上的配置文件默认加载路径
        # default="/home/og/XieSH/NCL/config/PMID2019/PMID2019_NCL.yaml",  # PMID2019专用，参数的默认值，改成自己服务器上的配置文件默认加载路径
        type=str,  # 参数的数据类型，为字符串类型
    )
    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",  # --ar参数用于是否使用自动恢复
        type=ast.literal_eval,
        dest='auto_resume',
        required=False,
        default=False,
    )

    parser.add_argument(
        "--local_rank",
        help='local_rank for distributed training',  # --local_rank参数用于分布式训练
        type=int,  # 整数类型
        default=0,  # 默认值是0，进程号为0，即默认使用第一块卡
    )

    parser.add_argument(
        "--model_dir",  # 模型存放目录
        type=str,
        default=None,
    )

    parser.add_argument(
        "opts",  # 位置参数，不能省略必须指定
        help="modify config options using the command-line",  # opts参数通过使用命令行修改配置文件选项
        default=None,
        # nargs入参把多个命令行参数与一个行为关联起来
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'  # 调试用
    if torch.cuda.is_available():
        print('using GPUS:%d' % torch.cuda.device_count())  # 打印服务器GPU数量
    else:
        print('no GPU')

    args = parse_args()  # 解析配置文件参数
    local_rank = args.local_rank
    rank = local_rank
    update_config(cfg, args)  # 更新配置参数对象cfg的值

    logger, log_file = create_logger(cfg, local_rank)  # 获取日志记录器和日志文件存放路径
    warnings.filterwarnings("ignore")
    auto_resume = args.auto_resume  # 自动恢复关闭

    setup_seed(cfg.RAND_SEED)  # 设置随机数种子

    # create model&log saving path 创建模型&日志保存
    if args.model_dir == None:  # 如果模型存放路径为空，则创建
        # 训练好的模型存放绝对路径
        model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models",
                                 str(datetime.now().strftime("%Y-%m-%d-%H-%M")))
    else:  # 如果模型存放路径不为空，则按配置文件中的来
        model_dir = args.model_dir
    code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "codes",
                            str(datetime.now().strftime("%Y-%m-%d-%H-%M")))  # 代码存放路径？没懂是存放什么代码
    tensorboard_dir = (
        os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboard",
                     str(datetime.now().strftime("%Y-%m-%d-%H-%M")))
        if cfg.TRAIN.TENSORBOARD.ENABLE  # 如果配置文件中打开了tensorboard，就设置tensorboard的存储路径
        else None
    )

    if local_rank == 0:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )
            if not click.confirm(
                    "\033[1;31;40mContinue and override the former directory?\033[0m",
                    default=False,
            ):
                exit(0)
            if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
                shutil.rmtree(tensorboard_dir)
        print("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)  # 训练函数所在文件目录
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )

    # DDP init  分布式并行训练初始化
    if cfg.TRAIN.DISTRIBUTED:  # 只有DDP开了才会执行
        if local_rank == 0:
            print('Init the process group for distributed training')
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        if local_rank == 0:
            print('Init complete')

    train_set = eval(cfg.DATASET.DATASET)("train", cfg)  # eval成数据集的类名，实例化数据集
    valid_set = eval(cfg.DATASET.DATASET)("valid", cfg)  # 实例化验证集

    annotations = train_set.get_annotations()  # 获取训练集标注
    num_classes = train_set.get_num_classes()  # 获取训练集类别数量
    device = torch.device("cuda")  # 指定GPU

    # 获取每类样本数的列表，获取所有样本标签构成的列表
    num_class_list, cat_list = get_category_list(annotations, num_classes, cfg)

    para_dict = {  # 把上面获取到的一些变量存放在参数字典中
        "num_classes": num_classes,  # 类别总数
        "num_class_list": num_class_list,  # 每类样本数量的列表
        "cfg": cfg,  # 配置文件
        "device": device,  # GPU
    }

    # 导入NIL和NBOD模块的损失函数
    criterion = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict)

    epoch_number = cfg.TRAIN.MAX_EPOCH  # 从配置文件中读取训练epoch数400轮

    # ----- BEGIN MODEL BUILDER ----- ----- 开始搭建模型 -----
    model = get_multi_model_final(cfg, num_classes, num_class_list, device, logger)  # backbone是ResNet32，然后GAP层，最后分类器是FC
    combiner = Combiner(cfg, device, num_class_list)  # 实例化组合器
    optimizer = get_optimizer(cfg, model)  # 获取优化器：SGD
    scheduler = get_scheduler(cfg, optimizer)  # 获取调度器：

    if cfg.TRAIN.DISTRIBUTED:  # DDP训练进
        model = model.cuda()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:  # 非DDP训练进：利用DataParallel()进行多卡计算
        model = model.cuda()  # 把模型放进GPU
        # model = torch.nn.DataParallel(model)    # 利用DataParallel()进行多卡计算
        model = torch.nn.DataParallel(model, device_ids=[0])  # 调试用：利用DataParallel()进行多卡计算
    # ----- END MODEL BUILDER ----- # 模型搭建结束

    if cfg.TRAIN.DISTRIBUTED:  # DDP训练进
        train_sampler = torch.utils.data.DistributedSampler(train_set)

        trainLoader = DataLoader(
            train_set,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )

        validLoader = DataLoader(
            valid_set,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TEST.NUM_WORKERS,
            pin_memory=True
        )

    else:  # 非DDP训练进

        trainLoader = DataLoader(  # 实例化训练集DataLoader
            train_set,  # 指定训练集
            batch_size=cfg.TRAIN.BATCH_SIZE,  # 指定batch_size：4
            shuffle=cfg.TRAIN.SHUFFLE,  # 指定打乱训练集：True
            num_workers=cfg.TRAIN.NUM_WORKERS,  # 指定工作线程：0
            pin_memory=cfg.PIN_MEMORY,  # 不知道是什么？
            drop_last=True  # 不懂是什么？
        )
        validLoader = DataLoader(  # 实例化验证集DataLoader
            valid_set,
            batch_size=cfg.TEST.BATCH_SIZE,  # 验证集batch_size=256
            shuffle=False,  # 验证集不打乱
            num_workers=cfg.TEST.NUM_WORKERS,  # 验证集工作线程：8
            pin_memory=cfg.PIN_MEMORY,
        )

    if tensorboard_dir is not None and local_rank == 0:
        dummy_input = torch.rand((1, 3) + cfg.INPUT_SIZE).to(device)  # 创建虚拟输入：一张CIFAR图片
        writer = SummaryWriter(log_dir=tensorboard_dir)  # 日志记录器
    else:
        writer = None

    best_result, best_epoch, start_epoch = 0, 0, 1  # 设立最佳结果、最佳epoch、开始epoch
    best_result_single, best_epoch_single = 0, 0  # 单专家最好结果、单专家最好的epoch数

    # # ----- BEGIN RESUME --------- 开始恢复？恢复什么
    all_models = os.listdir(model_dir)
    if len(all_models) <= 1 or auto_resume == False:
        auto_resume = False
    else:
        all_models.remove("best_model.pth")
        resume_epoch = max([int(name.split(".")[0].split("_")[-1]) for name in all_models])
        resume_model_path = os.path.join(model_dir, "epoch_{}.pth".format(resume_epoch))

    if cfg.RESUME_MODEL != "" or auto_resume:
        if cfg.RESUME_MODEL == "":
            resume_model = resume_model_path
        else:
            resume_model = cfg.RESUME_MODEL if '/' in cfg.RESUME_MODEL else os.path.join(model_dir, cfg.RESUME_MODEL)
        logger.info("Loading checkpoint from {}...".format(resume_model))
        checkpoint = torch.load(
            resume_model, map_location="cpu" if cfg.TRAIN.DISTRIBUTED else "cuda"
        )
        model.module.load_model(resume_model)
        if cfg.RESUME_MODE != "state_dict":
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_result = checkpoint['best_result']
            best_epoch = checkpoint['best_epoch']
    # ----- END RESUME ---------

    if rank == 0:
        logger.info(
            "-------------------Train start :{}  {}  {}-------------------".format(
                cfg.BACKBONE.TYPE, cfg.MODULE.TYPE, cfg.TRAIN.COMBINER.TYPE
            )
        )

    for epoch in tqdm(range(start_epoch, epoch_number + 1)):  # 400个epoch循环

        if cfg.TRAIN.DISTRIBUTED:  # DDP的采样器设置，暂时关闭不用
            train_sampler.set_epoch(epoch)

        scheduler.step()  # 完成一个epoch后，调用学习率调度器，调整学习率
        train_acc, train_loss = multi_networks_train_model(  # 调用模型训练器，开始训练
            trainLoader,  # 传入训练集DataLoader
            model,  # 传入模型
            epoch,  # 传入目前是第几轮epoch
            epoch_number,  # 传入要训练epoch总数：400
            optimizer,  # 传入优化器
            combiner,  # 传入组合器
            criterion,  # 传入损失函数
            cfg,  # 传入配置文件参数
            logger,  # 传入日志记录器
            writer=writer,  # 日志写入器
            rank=local_rank,
        )
        model_save_path = os.path.join(
            model_dir,
            "epoch_{}.pth".format(epoch),
        )
        if epoch % cfg.SAVE_STEP == 0 and local_rank == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_result': best_result,
                'best_epoch': best_epoch,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, model_save_path)

        loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}
        if cfg.VALID_STEP != -1 and epoch % cfg.VALID_STEP == 0:
            valid_acc, valid_loss, valid_acc_single = multi_network_valid_model_final(
                validLoader, epoch, model, cfg, criterion, logger, device,
                rank=rank, distributed=cfg.TRAIN.DISTRIBUTED, writer=writer
            )
            loss_dict["valid_loss"], acc_dict["valid_acc"] = valid_loss, valid_acc
            if epoch % cfg.SAVE_STEP == 0 and local_rank == 0:
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, model_save_path)
            if valid_acc > best_result and local_rank == 0:
                best_result, best_epoch = valid_acc, epoch
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "best_ensemble_model.pth")
                )
            if valid_acc_single > best_result_single and local_rank == 0:
                best_result_single, best_epoch_single = valid_acc_single, epoch
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "best_single_model.pth")
                )
            if rank == 0:
                logger.info(
                    "--------------Best_ensemble_Epoch:{:>3d}    Best_ensemble_Acc:{:>5.2f}%--------------".format(
                        best_epoch, best_result * 100
                    )
                )
                logger.info(
                    "--------------Best_single_Epoch:{:>3d}    Best_single_Acc:{:>5.2f}%--------------".format(
                        best_epoch_single, best_result_single * 100
                    )
                )

        if cfg.TRAIN.TENSORBOARD.ENABLE and local_rank == 0:
            writer.add_scalars("scalar/acc", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)
    if cfg.TRAIN.TENSORBOARD.ENABLE and local_rank == 0:
        writer.close()
    if rank == 0:
        logger.info(
            "-------------------Train Finished :{}~{}~{}~-------------------".format(cfg.NAME, best_result * 100,
                                                                                     best_result_single * 100)
        )
