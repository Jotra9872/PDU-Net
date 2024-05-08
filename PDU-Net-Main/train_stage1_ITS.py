import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from util.common import AverageMeter, pad_img
from util.scheduler import CosineScheduler
from dataset.loader import PairLoader
from models.PDUNet1 import Dehaze1
import random
import os

# data_path:   新服务器 /media/jiangaiwen/DATA_Wu/Dehaze  旧服务器 /media/jiangaiwen/DATA_OLD/Dehaze  自己电脑 G:/Dehaze
parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', default='16', type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')   # 半精度加速训练
parser.add_argument('--save_path', default='./saved_models_stage1/', type=str, help='path to models saving')  # 模型保存路径
parser.add_argument('--data_path', default='G:/Dehaze', type=str, help='path to dataset')  # 这个是dataset的路径(大路径) 和下面的train_set合起来就是训练集的路径
parser.add_argument('--log_path', default='./logs/', type=str, help='path to logs')    # log日志保存位置(大路径) 和下面的train_data，model合起来
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPUs used for training')

# ---------训练不同数据集需要修改的:
parser.add_argument('--train_path', default='RESIDE/ITS/', type=str, help='train dataset name')  # 训练集目标路径
parser.add_argument('--val_path', default='RESIDE/SOTS/SOTS-IN/', type=str, help='valid dataset name') # 测试集目标路径
parser.add_argument('--train_data', default='RESIDE_ITS', type=str, help='experiment setting')  # 保存日志的时候用到的,到底是用到的哪个数据集 RESIDE_ITS  RESIDE_OTS

# ---------训练不同的模型需要修改的:  注：首先需要修改的就是 from models.DehazeNet1_1 import Dehaze 改成DehazeNet2
parser.add_argument('--model', default='PDUNet1', type=str, help='model name')  # 保存日志的时候用到的，到底训练的是哪个模型
parser.add_argument('--Last_train_model_path', default='saved_models_stage1/RESIDE_ITS/PDUNet1/PDUNet1_934_39.4414.pth', help="path to continue training")
parser.add_argument('--train_flag', default=True, help="Choose to train from zero or continue training")  # 选择是重新训练还是 继续训练

args = parser.parse_args()
setting = {'batch_size': 8,
           'patch_size': 256,
           'valid_mode': "test",
           'edge_decay': 0.1,
           'only_h_flip': False,
           'optimizer': "adamw",
           'lr': 4e-4,  # 64--8e-4  128--16e-4  32--4e-4
           'eval_freq': 1,
           'weight_decay': 0.01,
           'epochs': 1500,
           'warmup_epochs': 50,
           'const_epochs': 0,
           'frozen_epochs': 200,
           }

def seed_torch(seed=3407):
    # random.seed(seed)   # Python的随机性
    # os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    # np.random.seed(seed)   # numpy的随机性
    # torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    # torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = True  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = False   # 选择确定性算法

seed_torch()

def train(train_loader, network, criterion, optimizer, scaler, frozen_bn=False):

    losses = AverageMeter()
    torch.cuda.empty_cache()
    network.eval() if frozen_bn else network.train()  # simplified implementation that other modules may be affected
    for batch in tqdm(train_loader):

        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with autocast(args.no_autocast):  # 当进入autocast上下文后，在这之后的cuda ops会把tensor的数据类型转换为半精度浮点型，从而在不损失训练精度的情况下加快运算
            output = network(source_img)
            loss = criterion(output, target_img)

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # 为了梯度放大
        scaler.step(optimizer)
        scaler.update()  # 准备着，看是否要增大scaler

        losses.update(loss.item())

    return losses.avg



def valid(val_loader, network):
    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in val_loader:

        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():  # torch.no_grad() may cause warning
            H, W = source_img.shape[2:]
            source_img = pad_img(source_img, network.module.patch_size if hasattr(network.module, 'patch_size') else 16)
            output = network(source_img).clamp_(-1, 1)
            output = output[:, :, :H, :W]

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg


if __name__ == '__main__':

    # define network 定义网络
    network = Dehaze1()
    network = nn.DataParallel(network).cuda()

    # define loss function 定义loss函数
    criterion = nn.L1Loss()

    # define optimizer 定义优化器
    optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'], weight_decay=setting['weight_decay'])

    lr_scheduler = CosineScheduler(optimizer, param_name='lr', t_max=setting['epochs'], value_min=setting['lr'] * 1e-2,
                                   warmup_t=setting['warmup_epochs'], const_t=setting['const_epochs'])
    # 以余弦函数为周期，并在每个周期最大值时重新设置学习率。以初始学习率为最大学习率，以 2∗Tmax2*Tmax2∗Tmax 为周期，在一个周期内先下降，后上升。
    # t_max：一次学习率周期的迭代次数，即 T_max 个 epoch 之后重新设置学习率。
    # value_min：最小学习率，即在一个周期中，学习率最小会下降到 eta_min。 warmup_t 预热的epoch。 const_t：最后一个EPOCH。

    wd_scheduler = CosineScheduler(optimizer, param_name='weight_decay', t_max=setting['epochs'])  # seems not to work
    scaler = GradScaler()        # 动态调整学习率

    # load saved model 加载并保存模型
    if args.train_flag:
        print('==> Continue training')
        model_info = torch.load(args.Last_train_model_path)
        network.load_state_dict(model_info['state_dict'],strict=False)
        # optimizer.load_state_dict(model_info['optimizer'])
        #
        # lr_scheduler.load_state_dict(model_info['lr_scheduler'])
        # wd_scheduler.load_state_dict(model_info['wd_scheduler'])
        # scaler.load_state_dict(model_info['scaler'])

        cur_epoch = model_info['cur_epoch']
        best_psnr = model_info['best_psnr']
        print('==> Load successfully ,The current epoch is:%d, current best_psnr is:%f ' % (cur_epoch, best_psnr))
    else:
        print('==> Not loaded, make a fresh start')
        best_psnr = 0
        cur_epoch = 0

    # define dataset 定义数据库
    train_dataset = PairLoader(os.path.join(args.data_path, args.train_path), 'train',
                               setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])

    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,  # shuffle 是否打乱加载数据
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(os.path.join(args.data_path, args.val_path), 'test',  # 'valid_mode':test
                             setting['patch_size'])

    val_loader = DataLoader(val_dataset,
                            batch_size=20,
                            num_workers=args.num_workers,
                            pin_memory=True)

    save_path = os.path.join(args.save_path, args.train_data, args.model)  # 保存训练完模型的位置

    os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(args.log_path, args.train_data, args.model))
    txt_name = args.model + str('.txt')
    txt_path = os.path.join(args.log_path, args.train_data, args.model, txt_name)
    # TensorBoard 中的SummaryWriter will record the loss and evaluation performance during training.
    #  SummaryWriter保存位置'./logs/RESIDE_ITS/DehazeNet'
    print('==> Start training, The current training dataset is ' + args.train_data)

    # start training 开始训练
    for epoch in range(cur_epoch, setting['epochs'] + 1):
        loss = train(train_loader, network, criterion, optimizer, scaler)
        lr_scheduler.step(epoch + 1)
        wd_scheduler.step(epoch + 1)
        writer.add_scalar('train_loss', loss, epoch)
        # scheduler.step（）按照Pytorch的定义是用来更新优化器的学习率的，一般是按照epoch为单位进行更换，
        # 即多少个epoch后更换一次学习率，因而scheduler.step()放在epoch这个大循环下。

        if epoch % setting['eval_freq'] == 0:
            # evaluate frequency 评估频率
            avg_psnr = valid(val_loader, network)
            writer.add_scalar('valid_psnr', avg_psnr, epoch)

            # 第一个参数：生成图像的名称 第二个参数：X轴的值 第三个参数：Y轴的值
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save({'cur_epoch': epoch + 1,
                            'best_psnr': best_psnr,
                            'state_dict': network.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'wd_scheduler': wd_scheduler.state_dict(),
                            'scaler': scaler.state_dict()},
                           os.path.join(save_path, args.model+"_"+str(epoch)+"_"+str(best_psnr)[:7]+'.pth')) # ./saved_models_stage1/  DehazeNet _ best_psnr(保留七位包括小数点) .pth

            writer.add_scalar('best_psnr', best_psnr, epoch)
            writer.add_scalar('valid_psnr', avg_psnr, epoch)
            print('epoch：%d\t The current epoch psnr：%f\t best psnr：%f\n' % (epoch, avg_psnr, best_psnr))
            out_txt = open(txt_path, 'a', encoding='utf-8')
            out_txt.write('epoch：%d\t The current epoch psnr：%f\t best psnr：%f\n' % (epoch, avg_psnr, best_psnr))
            out_txt.close()

# 在log终端打开
# tensorboard --logdir=RESIDE_ITS/DehazeNet1_4
