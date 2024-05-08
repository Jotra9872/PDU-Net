import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features # 找不到预先训练的就会下载
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(9):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 30):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)

        return [h_relu1, h_relu2, h_relu3]


class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg16().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [0.2, 0.5, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive

        return loss


class IdentityLoss(nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input_image, target_image):
        loss = self.criterion(input_image, target_image)
        return loss

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, input_image, output_image):
        dx_input = torch.abs(input_image[:, :, :-1, :-1] - input_image[:, :, :-1, 1:])
        dy_input = torch.abs(input_image[:, :, :-1, :-1] - input_image[:, :, 1:, :-1])
        dx_output = torch.abs(output_image[:, :, :-1, :-1] - output_image[:, :, :-1, 1:])
        dy_output = torch.abs(output_image[:, :, :-1, :-1] - output_image[:, :, 1:, :-1])
        tv_loss = torch.mean(dx_input) + torch.mean(dy_input) + torch.mean(dx_output) + torch.mean(dy_output)
        return tv_loss