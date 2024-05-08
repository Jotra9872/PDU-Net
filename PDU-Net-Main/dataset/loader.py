import os
import random
import numpy as np
import cv2
import torchvision
from torch.utils.data import Dataset

from util.common import hwc_to_chw, read_img
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision import datasets, transforms


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	# 图片进入augment 出来就是 256 256 3
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge 简单的边缘重设权重
	if random.random() < Hc / H * edge_decay:
		# random.random()用于生成一个0到1的随机符点数: 0 <= n < 1.0
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip 水平翻转
	if random.randint(0, 1) == 1:
		# andom.randint取到值的是闭区间   0<= <=1
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:  # 如果only_h_flip是False 就执行
		# bad data augmentations for outdoor 室外数据扩充错误
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs


class PairLoader(Dataset):  # 三个选择 训练集 训练集的GT 测试集
	def __init__(self, root_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		# self.root_dir = os.path.join(data_dir, sub_dir)
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1

		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}


class UnpairedLoader(Dataset):
	def __init__(self, root_dir, size=256, edge_decay=0, only_h_flip=False):
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		# self.root_dir = os.path.join(data_dir, sub_dir)
		self.root_dir = root_dir
		self.hazy_names = os.listdir(os.path.join(self.root_dir, 'hazy'))
		self.clear_names = os.listdir(os.path.join(self.root_dir, 'clear'))
		# random.shuffle(self.hazy_names)
		random.shuffle(self.clear_names)
		self.img_num = len(self.hazy_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		# print(self.hazy_names[idx])
		# print(self.clear_names[idx])
		# print("#"*10)

		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]

		hazy_img = read_img(os.path.join(self.root_dir, 'hazy', self.hazy_names[idx])) * 2 - 1
		clear_img = read_img(os.path.join(self.root_dir, 'clear', self.clear_names[idx])) * 2 - 1
		# [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)
		hazy_img = augment([hazy_img], self.size, self.edge_decay, self.only_h_flip)
		clear_img = augment([clear_img], self.size, self.edge_decay, self.only_h_flip)
		# 这里要分开augement  由于hazy 和clear 的图像大小并不一样 一同augment的话，会导致一张图片正确裁剪，另一张按相同的方法，并不能裁剪到256，256，所以要分开裁剪
		return {'hazy': hwc_to_chw(np.concatenate(hazy_img)), 'clear': hwc_to_chw(np.concatenate(clear_img))}


class DifferentNameLoader(Dataset):  # 三个选择 训练集 训练集的GT 测试集
	def __init__(self, root_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']
		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip
		# self.root_dir = os.path.join(data_dir, sub_dir)
		self.root_dir = root_dir
		self.GT_names = os.listdir(os.path.join(self.root_dir, 'GT'))
		self.hazy_names = os.listdir(os.path.join(self.root_dir, 'hazy'))
		self.img_num = len(self.hazy_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		# print(self.hazy_names[idx])
		# print(self.GT_names[idx])
		# print("#"*10)
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.hazy_names[idx]
		source_img = read_img(os.path.join(self.root_dir, 'hazy', self.hazy_names[idx])) * 2 - 1
		target_img = read_img(os.path.join(self.root_dir, 'GT', self.GT_names[idx])) * 2 - 1

		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}


# if __name__ == '__main__':
# 	# a = SingleImgLoader(os.path.join('G:/Dehaze', 'Real_OTS'), 256, 0, False)
# 	a = DifferentNameLoader(os.path.join('G:/Dehaze', 'Haze4K/Test'), 'test')
# 	b = a.__getitem__(0)
# 	a.__getitem__(2)
# 	a.__getitem__(3)
# 	a.__getitem__(4)
# 	a.__getitem__(5)
# 	a.__getitem__(6)
# 	a.__getitem__(7)
# 	a.__getitem__(8)
# 	a.__getitem__(9)
# 	a.__getitem__(10)
# 	a.__getitem__(11)
# 	a.__getitem__(12)
# 	a.__getitem__(13)
# 	a.__getitem__(14)
# 	a.__getitem__(15)
# 	a.__getitem__(16)
# 	a.__getitem__(17)


# # source_img = read_img('train.png') * 2 - 1
# target_img = read_img('GT.png')* 2 - 1
# print(source_img.shape)
# print(target_img.shape)
# cv2.imshow("1", source_img)
#
# cv2.imshow("2", target_img)
#
# [source_img, target_img] = augment([source_img, target_img], 256, 0,  False)
#
# print(source_img.shape)
# print(target_img.shape)
# cv2.imshow("3", source_img)
#
# cv2.imshow("4", target_img)
# cv2.waitKey(0)

#--------结论
# 1. 每次经过augment 会随机截取图像的一部分大小为 256 256 不管原图像大小多少 随机裁剪出每一张图片中的256 256 的大小
# 2. 也就是说每次 augment -》 train_dataset -》 train_loader 经过dataloader打包生成的数据集 是一个每张图片裁剪出来的256 256 大小的一块 patch
# 3. 每次对 hazy和Gt 里面对应的图片裁剪 是裁剪的同一个位置的 patch

# train_dataset = PairLoader(r'D:\pthon_pj\python_test\DehazeFormer-main\datasets', 'train', 'train', 256, 0, False)
# train_loader = DataLoader(train_dataset,
# 						  batch_size=1,
# 						  shuffle=True,  # shuffle 是否打乱加载数据
# 						  num_workers=0,
# 						  pin_memory=True,
# 						  drop_last=True)
#
# transforms.Normalize([0.5], [0.5])
# i = 0
# for batch in train_loader:
# 	i+=1
# 	print(i)
# 	source_img = batch['source']
# 	target_img = batch['target']
# 	img1 = torchvision.utils.make_grid(source_img) # 将一个批次的图片弄成网格模式 channel,h,w
# 	img1 = img1.numpy().transpose(1, 2, 0)
#
# 	img2 = torchvision.utils.make_grid(target_img)  # 将一个批次的图片弄成网格模式 channel,h,w
# 	img2 = img2.numpy().transpose(1, 2, 0)
# 	std = [0.5, 0.5, 0.5]
# 	mean = [0.5, 0.5, 0.5]
#
# 	img1 = img1 * std + mean # 去标准化
# 	img2 = img2 * std + mean
#
# 	cv2.imshow("img1", img1)
#
# 	cv2.imshow("img2", img2)
# 	cv2.waitKey(0)
#
# cv2.imshow("handwriting", img)
# cv2.waitKey(0)
