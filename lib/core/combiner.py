import torch, math
from core.evaluate import accuracy
from torch.nn import functional as F
from net.MOCO import shuffle_BN, shuffle_BN_DDP, unshuffle_BN_DDP, unshuffle_BN


class Combiner:  # 组合器
    def __init__(self, cfg, device, num_class_list=None):  # 入参:配置文件参数cfg、GPU、每类样本数{list:100}
        self.cfg = cfg  # 配置文件参数cfg
        self.type = cfg.TRAIN.COMBINER.TYPE  # 训练结合模式：默认
        self.device = device  # GPU
        self.num_class_list = torch.FloatTensor(num_class_list)  # 每类样本数{list:100}转换成{Tensor:(100,)}
        self.epoch_number = cfg.TRAIN.MAX_EPOCH  # 训练epoch：400
        self.initilize_all_parameters()  # 初始化所有参数

    def initilize_all_parameters(self):  # 初始化所有参数

        self.CON_ratio = self.cfg.LOSS.CON_RATIO  # 没懂？0.0
        self.show_step = self.cfg.SHOW_STEP  # 每100个mini-batch打印一次训练实况
        self.distributed = self.cfg.TRAIN.DISTRIBUTED  # 是否DDP？否

        print('_' * 100)  # 分割线
        print('combiner type: ', self.type)  # 打印训练组合模式：默认
        print('_' * 100)  # 分割线

    def update(self, epoch):
        self.epoch = epoch

    def forward(self, model, criterion, image, label, meta, **kwargs):
        return eval("self.{}".format(self.type))(  # 调用下面的方法：multi_network_default()
            model, criterion, image, label, meta, **kwargs
        )

    def multi_network_default(self, model, criterion, image, label, meta, **kwargs):

        for i in range(len(image)):  # 循环3次
            image[i], label[i] = image[i].to(self.device), label[i].to(self.device)  # 把三份图片和三份标签都传入GPU

        feature = model(image, feature_flag=True, label=label)  # 送入模型提取特征
        output = model(feature, classifier_flag=True)

        loss = criterion(output, label)

        average_result = torch.mean(torch.stack(output), dim=0)
        now_result = torch.argmax(average_result, 1)
        now_acc = accuracy(now_result.cpu().numpy(), label[0].cpu().numpy())[0]

        return loss, now_acc

    def multi_network_default_CON(self, model, criterion, image, label, meta, **kwargs):

        image_p = []
        image_k = []
        for i in range(len(image)):
            image_p.append(image[i][0].to(self.device))
            image_k.append(image[i][1].to(self.device))
            label[i] = label[i].to(self.device)

        # shuffle BN
        if self.distributed:
            image_k, idx_unshuffle = shuffle_BN_DDP(image_k)
            pass
        else:
            image_k, idx_unshuffle = shuffle_BN(image_k)  # 对image_K里的图片进行shuffle

        feature = model((image_p, image_k), feature_flag=True, label=label)  # 返回2个特征输出，一个正常的，一个用于自监督学习的
        output_ce, output_p, output_k = model(feature, classifier_flag=True)  # 返回正常模型形状为(64, 100)、(64, 64)的logits和自监督模型形状为(64, 64)的logits

        # unshuffle
        if self.distributed:
            output_k = unshuffle_BN_DDP(output_k, idx_unshuffle)
        else:
            output_k = unshuffle_BN(output_k, idx_unshuffle)  # 逆shuffle操作

        loss_ce = criterion(output_ce, label, feature=feature, classifier=model.module.classifier)  # 计算NIL和NBOD模块的全部类别和困难类别的损失

        average_result = torch.mean(torch.stack(output_ce), dim=0)  # 对3个专家输出形状为(64, 100)的logits求均值
        now_result = torch.argmax(average_result, 1)  # 获取logits最大的下标
        now_acc = accuracy(now_result.cpu().numpy(), label[0].cpu().numpy())[0]  # 计算准确率

        # contrastive_loss
        loss_CON = 0  # 对比学习损失
        for i, (q, k) in enumerate(zip(output_p, output_k)):
            q = F.normalize(q, dim=1)
            k = F.normalize(k, dim=1)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # 没太看懂
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, model.module.MOCO[i].queue.clone().detach()])  # 没太看懂

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)  # 把正例和负例logits拼接起来

            # apply temperature
            logits /= model.module.MOCO[i].T  # 除以论文中对比学习损失Lcon的温度系数τ

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()  # bs个0值

            # dequeue and enqueue
            if self.distributed:
                model.module.MOCO[i]._dequeue_and_enqueue_DDP(k)
            else:
                model.module.MOCO[i]._dequeue_and_enqueue(k)  # 出队和进队

            loss_CON += F.cross_entropy(logits, labels)  # 计算对比学习损失

        loss = loss_ce + loss_CON * self.CON_ratio  # 计算总损失值

        return loss, now_acc  # 返回总损失值和准确率
