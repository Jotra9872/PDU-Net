import cv2
import torch
import torch.nn as nn
import numpy as np


def get_dark_channel(I, w):  # 求暗通道

    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    dc = maxpool(0 - I[:, :, :, :])
    return -dc


def get_bright_channel(I, w):  # 求亮通道
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    bc = maxpool(I[:, :, :, :])
    return bc


def get_atmosphere1(I, dark_ch, p):  # 暗通道 求大气光
    B, _, H, W = dark_ch.shape
    num_pixel = int(p * H * W)
    flat_dc = dark_ch.reshape(B, H * W)
    flat_I = I.reshape(B, 3, H * W)
    index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]  # descending控制是否降序,默认为False
    A = torch.zeros((B, 3)).to('cuda')

    for i in range(B):
        # A[i] = flat_I[i, :, index].mean((1, 2))
        A[i] = flat_I[i, :, index[i][torch.argsort(torch.max(flat_I[i][:, index[i]], 0)[0], descending=True)[0]]]
    return A


def get_atmosphere2(I, bright_ch, p):  # 亮通道 求大气光
    B, _, H, W = bright_ch.shape
    num_pixel = int(p * H * W)
    flat_bc = bright_ch.reshape(B, H * W)
    flat_I = I.reshape(B, 3, H * W)
    index = torch.argsort(flat_bc, descending=False)[:, :num_pixel]
    A = torch.zeros((B, 3)).to('cuda')

    for i in range(B):
        A[i] = flat_I[i, :, index].mean((1, 2))  # A[i] = flat_I[i, :, index].mean((1, 2))

    return A


def dark_channel_transmission(img, w=15, p=0.0001):   # 利用暗通道先验求解传输图
    dc = get_dark_channel(img, w)
    A = get_atmosphere1(img, dc, p)
    norm_I = (1 - img) / (1 - A[:, :, None, None] + 1e-6)
    dc = get_dark_channel(norm_I, w)  # 这一步 才真正得到暗通道图
    t = (1 - 0.95 * dc)  # w 控制去雾程度 这里取0.95

    return t


def bright_channel_transmission(img, w=15, p=0.0001):  # 利用亮通道先验求解传输图

    bc = get_bright_channel(img, w)
    A = get_atmosphere2(img, bc, p)
    norm_I = (1 - img) / (1 - A[:, :, None, None] + 1e-6)
    bc = get_bright_channel(norm_I, w)
    t = (1 - 0.95 * bc)

    return t

def bright_channel_transmission2(img, w=15, p=0.0001):  # 利用亮通道先验求解传输图

    bc = get_bright_channel(img, w)
    A = get_atmosphere1(img, bc, p)
    norm_I = (1 - img) / (1 - A[:, :, None, None] + 1e-6)
    bc = get_bright_channel(norm_I, w)
    t = (1 - 0.95 * bc)

    return t



def dark_bright_transmission(img, w=15, p=0.0001):  # w=1代表窗口大小，P=0.0001代表取0.1%最亮点
    dc = get_dark_channel(img, w)  # 求得暗通道
    bc = get_bright_channel(img, w)  # 求得亮通道

    A1 = get_atmosphere1(img, dc, p)  # 得暗通道大气光
    A2 = get_atmosphere2(img, bc, p)  # 得亮通道大气光
    A = 0.75 * A1 + 0.25 * A2  # 求得混合暗通道

    norm_I = (1 - img) / (1 - A[:, :, None, None] + 1e-6)
    channel = get_dark_channel(norm_I, w)  # 算两次 dark_channel 是因为 需要求两次最小值

    t = (1 - 0.95 * channel)
    return t


# 将获得的混合通道先验图和原图concat
class Concat(nn.Module):
    def __init__(self, in_chans=3, out_chans=4):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

    def forward(self, x):
        t = dark_bright_transmission(x, w=15, p=0.0001)
        x = torch.cat((x, t), 1)

        return x

# if __name__ == '__main__':
#     net = Concat().to('cuda:0')
#     input = torch.rand(4, 3, 256, 256).to('cuda:0')
#     output = net(input)
#     print(output.shape)


if __name__ == '__main__':
    # path = r'G:\1.png'
    path = r'D:\Python_Project\PDU-Net-Main\1.png'
    img1 = cv2.imread(path, 1)
    print(img1.shape)
    cv2.imshow('1', img1)
    cv2.waitKey(0)

    input5 = torch.Tensor(img1).to('cuda:0')
    input5 = input5.permute(2, 0, 1).unsqueeze(0)
# dark_bright_transmission bright_channel_transmission   dark_channel_transmission

    output5 = bright_channel_transmission2(input5, w=5)
    output5 = torch.squeeze(output5, dim=0).permute(1, 2, 0)
    img5 = output5.detach().cpu().numpy()

    cv2.imshow('5', img5)
    cv2.waitKey(0)