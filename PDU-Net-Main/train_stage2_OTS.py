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

from util.common import AverageMeter, pad_img
from util.scheduler import CosineScheduler
from dataset.loader import PairLoader, UnpairedLoader
from dataset.concat_dataset import ConcatDataset
from models.PDUNet1 import Dehaze1
from models.PDUNet2_2 import Dehaze2
from models.Losses import ContrastLoss, IdentityLoss, TVLoss


# data_path:   新服务器 /media/jiangaiwen/DATA_Wu/Dehaze  旧服务器 /media/jiangaiwen/DATA_OLD/Dehaze  自己电脑 G:/Dehaze
parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', default='8', type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')     # 半精度加速训练
parser.add_argument('--save_path', default='./saved_models_stage2/', type=str, help='path to models saving')  # 模型保存路径
parser.add_argument('--data_path', default='G:/Dehaze', type=str, help='path to dataset')  # 这个是dataset的路径(大路径) 和下面的train_set合起来就是训练集的路径
parser.add_argument('--log_path', default='./logs/', type=str, help='path to logs')    # log日志保存位置(大路径) 和下面的train_data，model合起来
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPUs used for training')


# ---------训练不同数据集需要修改的:
parser.add_argument('--unlabel_train_path', default='Real_OTS', type=str, help='train dataset name')  # 真实有雾训练集目标路径
parser.add_argument('--label_train_path', default='RESIDE/OTS/', type=str, help='train dataset name')  # 合成有雾训练集目标路径

parser.add_argument('--val_path', default='RESIDE/SOTS/SOTS-OUT', type=str, help='valid dataset name')  # 测试集目标路径
parser.add_argument('--train_data', default='Real_OTS', type=str, help='experiment setting')  # 保存日志的时候用到的,到底是用到的哪个数据集 RESIDE_ITS  RESIDE_OTS

# ---------训练不同的模型需要修改的:  注：首先需要修改的就是 from models.DehazeNet1_1 import Dehaze 改成DehazeNet2
parser.add_argument('--model', default='PDUNet2_2', type=str, help='model name')  # 保存日志的时候用到的，到底训练的是哪个模型
parser.add_argument('--Last_train_model1_path', default='saved_models_stage1/RESIDE_OTS/PDUNet1/PDUNet1_35_34.9182.pth', help="path to continue training")
parser.add_argument('--Last_train_model2_path', default='', help="path to continue training")
parser.add_argument('--train_flag', default=False, help="Choose to train from zero or continue training")  # 选择是重新训练还是 继续训练

# ----------对比学习用到的
parser.add_argument('--is_ab', type=bool, default=False)

args = parser.parse_args()
setting = {'batch_size': 4,
           'patch_size': 256,
           'valid_mode': "test",
           'edge_decay': 0,
           'only_h_flip': False,
           'eval_freq': 1,
           'optimizer': "adamw",  # adamw,adam
           'lr': 1e-4,
           'weight_decay': 0.01,
           'epochs': 1000,
           'l_crloss_weight':1,
           'lwf_loss_weight':1,
           'lide_loss_weight':1
           }


def train(train_loader, network1, network2, Cr_loss, identity_loss, Tv_loss, optimizer, scaler, frozen_bn=False):


    losses = AverageMeter()
    torch.cuda.empty_cache()
    network1.eval()  # 每个epoch前都定义下这一块 防止network1偷偷开启 update
    network2.eval() if frozen_bn else network2.train()  # .eval不启用BatchNormalization 和 Dropout 而.train启用 BatchNormalization 和 Dropout
    for batch in tqdm(train_loader):

        # --- load data --- #

        label_haze = batch[0]['source'].cuda()
        label_gt = batch[0]['target'].cuda()

        unlabel_haze = batch[1]['hazy'].cuda()
        unlabel_gt = batch[1]['clear'].cuda()

        with autocast(args.no_autocast):   # 当进入autocast上下文后，在这之后的cuda ops会把tensor的数据类型转换为半精度浮点型，从而在不损失训练精度的情况下加快运算

            out_1_label = network1(label_haze)
            out_1_unlabel = network1(unlabel_haze)

            out_2_label = network2(label_haze)
            out_2_unlabel = network2(unlabel_haze)

            # 对比学习loss
            l_crloss = Cr_loss(out_2_unlabel,  unlabel_gt, unlabel_haze)   # out为输出去雾结果 clear_img无雾图 ，hazy_img有雾图
            # 无遗忘loss
            lwf_loss_label = F.smooth_l1_loss(out_2_label, out_1_label)
            lwf_loss_unlabel = F.smooth_l1_loss(out_2_unlabel, out_1_unlabel)
            lwf_loss = lwf_loss_label + lwf_loss_unlabel
            # 身份损失（identity loss）
            lide = identity_loss(unlabel_haze, out_2_unlabel)
            #  全变损失 Total Variation (TV)
            ltv = Tv_loss(unlabel_haze, out_2_unlabel)

            # loss = l_crloss + lwf_loss + lide + ltv
            # loss = loss1 + loss2 / (loss2 / loss1).detach() + loss3 / (loss3 / loss1).detach()
            loss = l_crloss + lwf_loss / (lwf_loss / l_crloss).detach() + lide / (lide / l_crloss).detach() + ltv / (ltv / l_crloss).detach()
            # l_crloss 9.53 lwf_loss 0.0070 lide0.1024 ltv0.1828
        optimizer.zero_grad()
        scaler.scale(loss).backward()  # 为了梯度放大
        scaler.step(optimizer)
        scaler.update()  # 准备着，看是否要增大scaler

        losses.update(loss.item())

    return losses.avg


def valid(val_loader, network1, network2):
    PSNR = AverageMeter()

    torch.cuda.empty_cache()

    network1.eval()
    network2.eval()
    for batch in val_loader:

        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():  # torch.no_grad() may cause warning
            H, W = source_img.shape[2:]
            source_img = pad_img(source_img, network2.module.patch_size if hasattr(network2.module, 'patch_size') else 16)

            output = network2(network1(source_img).clamp_(-1, 1))

            output = output[:, :, :H, :W]

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    return PSNR.avg


# 定义 冻结预训练模型的参数 将network2中的network1的所有参数都设为param.requires_grad = False使其不更新，
# 其余是network2新增的参数设为param.requires_grad = True  使其在微调时参与更新
# freeze_model的作用是使model中的to_freeze_dict参数 全部不参与个更新，其余参与更新

def freeze_model(model, to_freeze_dict):
    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model


if __name__ == '__main__':

    # define network 定义网络
    network1 = Dehaze1()
    network2 = Dehaze2()
    network1 = nn.DataParallel(network1).cuda()
    network2 = nn.DataParallel(network2).cuda()

    # define loss function 定义loss函数
    Cr_loss = ContrastLoss(ablation=args.is_ab)
    identity_loss = IdentityLoss()
    Tv_loss = TVLoss()
    #  load stage1_model 加载有监督训练的预训练stage1模型到network1
    pre_state_dict1 = torch.load(args.Last_train_model1_path)['state_dict']
    network1.load_state_dict(pre_state_dict1)
    print('==>Load stage1_model from %s success.' % args.Last_train_model1_path)
    network1 = freeze_model(model=network1, to_freeze_dict=pre_state_dict1)

    #  load stage1_model 加载有监督训练的预训练stage1模型到network2  并且冻结除了微调部分之外的所有参数
    network2.load_state_dict(pre_state_dict1, strict=False)
    print('==>Load stage2_model from %s success.' % args.Last_train_model1_path)
    network2 = freeze_model(model=network2, to_freeze_dict=pre_state_dict1)

    # 测试network2中的原参数是否被冻结
    # for name, param in network2.named_parameters():
    #     if "zero" in name:
    #         print(name)
    #         print(param.requires_grad)
    #     else:
    #         print(name)
    #         print(param.requires_grad)    # requires_grad False代表不参与更新 True 代表参与更新
    # define optimizer 定义优化器  并且设置filter过滤器 只对 网络中requires_grad = true 的参数进行更新
    if setting['optimizer'] == 'adam':
        optimizer_network2 = torch.optim.Adam(filter(lambda p: p.requires_grad, network2.parameters()), lr=setting['lr'])
    elif setting['optimizer'] == 'adamw':
        optimizer_network2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, network2.parameters()), lr=setting['lr'])
    else:
        raise Exception("ERROR: unsupported optimizer")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_network2, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
    scaler = GradScaler()        # 动态调整学习率

    # define dataset 定义数据库加载数据

    train_label_dataset = PairLoader(os.path.join(args.data_path, args.label_train_path), 'train',
                                     setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])  # 合成有雾,用于添加防止灾难性遗忘Loss

    train_unlabel_dataset = UnpairedLoader(os.path.join(args.data_path, args.unlabel_train_path),
                                          setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])  # 真实有雾

    train_loader = DataLoader(ConcatDataset(train_label_dataset, train_unlabel_dataset),
                              batch_size=setting['batch_size'],
                              shuffle=True,  # shuffle 是否打乱加载数据
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    val_dataset = PairLoader(os.path.join(args.data_path, args.val_path), setting['valid_mode'],
                             setting['patch_size'])

    val_loader = DataLoader(val_dataset,
                            batch_size=1,  # 由于OTS中存在不一样的大小图片 在每一轮测试的时候 batch_size要设为1
                            num_workers=args.num_workers,
                            pin_memory=True)
    save_path = os.path.join(args.save_path, args.train_data, args.model)  # 保存训练完模型的位置
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_path, args.train_data, args.model))
    txt_name = args.model + str('.txt')
    txt_path = os.path.join(args.log_path, args.train_data, args.model, txt_name)
    print('==> Start training, The current training dataset is ' + args.train_data)
    loss1 = 1000

    # start training 开始训练
    for epoch in range(0, setting['epochs'] + 1):

        loss = train(train_loader, network1, network2, Cr_loss, identity_loss, Tv_loss, optimizer_network2, scaler)
        writer.add_scalar('train_loss', loss, epoch)
        print('The current loss is:%f' % loss)
        if epoch % setting['eval_freq'] == 0:
            cur_psnr = valid(val_loader, network1, network2)
            if loss < loss1:
                loss1 = loss
                torch.save({'cur_epoch': epoch + 1,
                            'state_dict': network2.state_dict(),
                            'optimizer': optimizer_network2.state_dict(),
                            'scaler': scaler.state_dict()},
                           os.path.join(save_path, args.model + "_" + str(epoch) + "_" + str(loss)[:7] + '.pth'))

            writer.add_scalar('current_psnr', cur_psnr, epoch)
            print('epoch：%d\t The current epoch psnr：%f\t The current loss：%f\n' % (epoch, cur_psnr, loss))
            out_txt = open(txt_path, 'a', encoding='utf-8')
            out_txt.write('epoch：%d\t The current epoch psnr：%f\t The current loss：%f\n' % (epoch, cur_psnr, loss))
            out_txt.close()
