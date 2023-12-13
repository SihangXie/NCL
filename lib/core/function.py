from core.evaluate import accuracy, AverageMeter
import torch
import time
import numpy as np


def multi_networks_train_model(  # 多专家网络训练器
        trainLoader, model, epoch, epoch_number, optimizer, combiner, criterion, cfg, logger, rank=0, **kwargs
):
    if cfg.EVAL_MODE:  # 验证阶段进
        model.eval()
    else:  # 训练阶段进
        model.train()

    network_num = len(cfg.BACKBONE.MULTI_NETWORK_TYPE)  # 专家数量：3
    trainLoader.dataset.update(epoch)  # 此方法可以实现渐进式调整训练集
    combiner.update(epoch)  # 此方法可以实现渐进式调整组合器
    criterion.update(epoch)  # 此方法可以实现渐进式调整损失函数

    start_time = time.time()  # 记录开始训练时间毫秒数
    number_batch = len(trainLoader)  # 计算一个epoch一共有多少批mini-batch：10847/4≈2711

    all_loss = AverageMeter()  # 实例化平均损失函数计算器
    acc = AverageMeter()  # 实例化准确率
    for i, (image, label, meta) in enumerate(trainLoader):  # 应该要循环2711次吧
        # 由于采用了自监督策略，此处image会得到2份图片，是同一批图片经过两个transform得来的
        image_list = [image] * network_num  # 图片batch复制成3份相同的{Tensor:(4,3,32,32)}的列表{list:3}
        label_list = [label] * network_num  # 标签batch同上复制成了3份{Tensor:(4,)}的列表{list:3}
        meta_list = [meta] * network_num  # 元数据batch复制成3份{dict:1}字典的列表{list:3}

        cnt = label_list[0].shape[0]  # 获取标签列表第0个批量张量维度的第0个维度：batch_size

        optimizer.zero_grad()  # 梯度清零
        loss, now_acc = combiner.forward(model, criterion, image_list, label_list, meta_list, now_epoch=epoch,
                                         train=True, cfg=cfg, iteration=i, log=logger,
                                         class_list=criterion.num_class_list)  # 把三份图片、三份标签、三份图片ID传入结合器的multi_network_default()方法

        if cfg.NETWORK.MOCO:  # CIFAR不进
            alpha = cfg.NETWORK.MA_MODEL_ALPHA  # α是什么？
            for net_id in range(network_num):
                net = ['backbone', 'module']
                for name in net:
                    for ema_param, param in zip(eval('model.module.' + name + '_MA').parameters(),
                                                eval('model.module.' + name).parameters()):
                        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)  # 指数滑动平均模型的参数保留0.999，新参数只保留0.001

        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        if i % cfg.SHOW_STEP == 0 and rank == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    if rank == 0:
        logger.info(pbar_str)
    return acc.avg, all_loss.avg


def multi_network_valid_model_final(
        dataLoader, epoch_number, model, cfg, criterion, logger, device, rank, **kwargs
):
    model.eval()
    network_num = len(cfg.BACKBONE.MULTI_NETWORK_TYPE)
    cnt_all = 0
    every_network_result = [0 for _ in range(network_num)]

    with torch.no_grad():
        all_loss = AverageMeter()
        acc_avg = AverageMeter()

        for i, (image, label, meta) in enumerate(dataLoader):

            image, label = image.to(device), label.to(device)
            image_list = [image for i in range(network_num)]

            if cfg.NETWORK.MOCO:
                feature = model((image_list, image_list), label=label, feature_flag=True)
                output_ce, output, output_MA = model(feature, classifier_flag=True)
            else:
                feature = model(image_list, label=label, feature_flag=True)
                output_ce = model(feature, classifier_flag=True)

            loss = criterion(output_ce, (label,))

            for j, logit in enumerate(output_ce):
                every_network_result[j] += torch.sum(torch.argmax(logit, dim=1).cpu() == label.cpu())

            average_result = torch.mean(torch.stack(output_ce), dim=0)
            now_result = torch.argmax(average_result, 1)

            acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            cnt_all += cnt
            all_loss.update(loss.data.item(), cnt)
            acc_avg.update(acc, cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_ensemble_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc_avg.avg * 100
        )
        if rank == 0:
            for i, result in enumerate(every_network_result):
                logger.info("network {} Valid_single_Acc: {:>5.2f}%".format(i, float(result) / cnt_all * 100))
            logger.info(pbar_str)
        best_single_acc = max(every_network_result) / cnt_all
    return acc_avg.avg, all_loss.avg, best_single_acc
