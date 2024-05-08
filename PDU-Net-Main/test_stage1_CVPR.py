import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from util.common import AverageMeter, write_img, chw_to_hwc
from dataset.loader import PairLoader
from models.PDUNet1 import Dehaze1
from metrics import ssim, psnr
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', default='16', type=int, help='number of workers')
parser.add_argument('--data_path', default='G:/Dehaze/', type=str, help='path to dataset')  # 这个是dataset的路径
parser.add_argument('--result_path', default='./results/', type=str, help='path to results saving')

# 测试不同模型需要修改的
parser.add_argument('--model', default='PDUNet1', type=str, help='model name')
# test_data: SOTS-IN SOTS-IN Dense-Haze NH-haze2
parser.add_argument('--test_data', default='SOTS-OUT', type=str, help='experiment setting')  # 选择测试是用哪个测试集  SOTS-IN SOTS-OUT
# val_path : RESIDE/SOTS/SOTS-OUT  RESIDE/SOTS/SOTS-IN Dense-Haze NH-haze2/Test
parser.add_argument('--val_path', default='RESIDE/SOTS/SOTS-OUT', type=str, help='valid dataset name')  # 测试集目标路径 SOTS-IN SOTS-OUT
parser.add_argument('--saved_model_path',default='saved_models_stage1/RESIDE_OTS/PDUNet1/PDUNet1_35_34.9182.pth', help="path to models saving")  # 需要加载模型的保存位置

args = parser.parse_args()


def test(test_loader,  network, result_path):
    network.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    network.eval()

    os.makedirs(os.path.join(result_path, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_path, 'results.csv'), 'w')

    for i, batch in enumerate(test_loader):
        inputs = batch['source'].cuda()
        targets = batch['target'].cuda()
        filename = batch['filename'][0]
        with torch.no_grad():
            output = network(inputs)

        pred = output.clamp(0, 1)
        gt = targets.clamp(0, 1)

        ssim1 = ssim(pred, gt).item()
        psnr1 = psnr(pred, gt)
        ssims.append(ssim1)
        psnrs.append(psnr1)

        print('Test: [{0}]\t'
              'PSNR: {psnr:.02f} \t'
              'SSIM: {ssim:.03f} '
              .format(i, psnr=psnr1, ssim=ssim1))

        f_result.write('%s,%.02f,%.03f\n' % (filename, ssim1, psnr1))
        output = output.clamp_(-1, 1) * 0.5 + 0.5
        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)
    f_result.close()
    ssim_eval = np.mean(ssims)
    psnr_eval = np.mean(psnrs)
    os.rename(os.path.join(result_path, 'results.csv'), os.path.join(result_path, f'{ssim_eval:.3f}_{psnr_eval:.4f}.csv'))
    return ssim_eval, psnr_eval



if __name__ == '__main__':
    # define network 定义网络
    network = Dehaze1()
    network = nn.DataParallel(network).cuda()

    # 定义准备加载模型的地址
    saved_model_path = args.saved_model_path

    if os.path.exists(saved_model_path):
        print('==> Start testing, current model name: ' + args.model)

        network.load_state_dict(torch.load(args.saved_model_path)['state_dict'])
        network.eval()
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_path, args.val_path)

    test_dataset = PairLoader(dataset_dir, 'test', 'test')  # F:/Dataset_All/DehazeFormer-main/data/RESIDE-IN
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    result_dir = os.path.join(args.result_path, args.test_data, args.model)
    ssim_eval, psnr_eval = test(test_loader, network, result_dir)
