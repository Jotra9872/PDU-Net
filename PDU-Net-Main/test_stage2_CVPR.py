import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
from util.common import AverageMeter, write_img, chw_to_hwc
from dataset.loader import DifferentNameLoader
from models.DehazeNet1_3 import Dehaze1
from models.DehazeNet2_1 import Dehaze2
from metrics import ssim, psnr
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', default='16', type=int, help='number of workers')
parser.add_argument('--data_path', default='G:/Dehaze/', type=str, help='path to dataset')  # 这个是dataset的路径
parser.add_argument('--result_path', default='./results/', type=str, help='path to results saving')

# 测试不同模型需要修改的
parser.add_argument('--model1', default='DehazeNet1_3', type=str, help='model name')
parser.add_argument('--model2', default='DehazeNet2_1', type=str, help='model name')
# test_data: SOTS-IN SOTS-IN Dense-Haze NH-haze2
parser.add_argument('--test_data', default='NH-haze', type=str, help='experiment setting')
# val_path : RESIDE/SOTS/SOTS-OUT  RESIDE/SOTS/SOTS-IN Dense-Haze NH-haze2/Test
parser.add_argument('--val_path', default='NH-haze', type=str, help='valid dataset name')
parser.add_argument('--saved_model1_path', default='saved_models_stage1/RESIDE_OTS/DehazeNet1_3/DehazeNet1_3_18_32.7505.pth', help="path to continue training")
parser.add_argument('--saved_model2_path', default='saved_models_stage2/Real_OTS/DehazeNet2_1/DehazeNet2_1_0_18.9963.pth', help="path to continue training")

args = parser.parse_args()


def test(test_loader,  network1, network2,result_path):
    network1.eval()
    network2.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    os.makedirs(os.path.join(result_path, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_path, 'results.csv'), 'w')

    for i, batch in enumerate(test_loader):
        inputs = batch['source'].cuda()
        targets = batch['target'].cuda()
        filename = batch['filename'][0]
        with torch.no_grad():
            output = network2(network1(inputs))

        ssim1 = ssim(output, targets).item()
        psnr1 = psnr(output, targets)
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
    network1 = Dehaze1()
    network2 = Dehaze2()
    network1 = nn.DataParallel(network1).cuda()
    network2 = nn.DataParallel(network2).cuda()

    # 定义准备加载模型的地址
    saved_model1_path = args.saved_model1_path
    saved_model2_path = args.saved_model2_path
    if os.path.exists(saved_model1_path) and os.path.exists(saved_model2_path):
        print(f'==> Start testing, current stage1_model name: {args.model1} current stage2_model name: {args.model2}')
        # load stage1_model 加载有监督训练的预训练stage1模型到network1
        network1_state_dict = torch.load(args.saved_model1_path)['state_dict']
        network1.load_state_dict(network1_state_dict, strict=False)
        network1.eval()
        # load stage1_model 加载无监督监督训练stage2模型到network2
        network2_state_dict = torch.load(args.saved_model2_path)['state_dict']
        network2.load_state_dict(network2_state_dict, strict=False)
        network2.eval()


    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_path, args.val_path)

    test_dataset = DifferentNameLoader(dataset_dir, 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    result_dir = os.path.join(args.result_path, args.test_data, args.model2)
    test(test_loader, network1,network2, result_dir)
