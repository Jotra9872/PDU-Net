import numpy as np
import cv2
import torch.nn.functional as F
import math
import torch
import torch.nn as nn


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
	"""Computes and stores the average and current values of a list计算并存储列表的平均值和当前值"""
	def __init__(self):
		self.len = 10000  # set up the maximum length
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len '
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			

def read_img(filename):
	img = cv2.imread(filename)
	# print(img)
	h = img.shape[0]
	w = img.shape[1]
	if h<256 or w<256:
		img = cv2.resize(img, dsize=(256, 256))

	return img[:, :, ::-1].astype('float32') / 255.0


def write_img(filename, img):
	img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
	cv2.imwrite(filename, img)


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).copy()

def pad_img(x, patch_size):
	_, _, h, w = x.size()
	mod_pad_h = (patch_size - h % patch_size) % patch_size
	mod_pad_w = (patch_size - w % patch_size) % patch_size
	x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
	return x

# ------------------------------------------------------------------
class GroupNorm32(nn.GroupNorm):
	def forward(self, x):
		return super().forward(x.float()).type(x.dtype)



def zero_module(module):
	"""
    Zero out the parameters of a module and return it.
    """
	for p in module.parameters():
		p.detach().zero_()
	return module


def scale_module(module, scale):
	"""
	Scale the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().mul_(scale)
	return module


def mean_flat(tensor):
	"""
    Take the mean over all non-batch dimensions.
    """
	return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
	"""
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: a nn.Module for normalization.
    """
	return GroupNorm32(32, channels)

