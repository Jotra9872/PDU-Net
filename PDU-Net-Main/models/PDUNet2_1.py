import torch.backends.cudnn
from models.DFBlock import *
from models.DcBcPrior import Concat
from util.common import *


class Dehaze2(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, *args, **kwargs):
        super(Dehaze2, self).__init__()
        # setting  设置
        self.patch_size = 4
        # concat img dark channel and bright channel prior
        self.concat_img_prior = Concat(in_chans=3, out_chans=4)
        #  split image into non-overlapping patches 将图像分割为不重叠的patch
        self.patch_embed = PatchEmbed(patch_size=1, in_chans=4, embed_dim=24, kernel_size=3)  # 把输入图像4变成通道数24 大小256
        self.fusion = SKFusion(96)
        # downsample
        self.down1 = PatchEmbed(patch_size=2, in_chans=24, embed_dim=48)  # x=(bs,24,256,256) -> x=(bs,48,128,128)
        self.down2 = PatchEmbed(patch_size=2, in_chans=48, embed_dim=96)  # x=(bs,48,128,128) -> x=(bs,96,64,64)
        # DF Block Block都不改变维度，输入输出维度保持一致
        self.Big_BLockA1 = BasicLayer1(network_depth=48, dim=96, depth=12,
                                       num_heads=4, mlp_ratio=2.,
                                       norm_layer=RLN, window_size=8,
                                       attn_ratio=1/4, attn_loc='last', conv_type='DWConv')
        self.Big_BLockA2 = BasicLayer1(network_depth=48, dim=96, depth=12,
                                       num_heads=4, mlp_ratio=4.,
                                       norm_layer=RLN, window_size=8,
                                       attn_ratio=1/2, attn_loc='last', conv_type='Conv')
        self.Big_BLockA3 = BasicLayer1(network_depth=48, dim=96, depth=12,
                                       num_heads=4, mlp_ratio=4.,
                                       norm_layer=RLN, window_size=8,
                                       attn_ratio=3/4, attn_loc='last', conv_type='DilConv')
        self.Big_BLockA4 = BasicLayer1(network_depth=48, dim=96, depth=6,
                                       num_heads=4, mlp_ratio=2.,
                                       norm_layer=RLN, window_size=8,
                                       attn_ratio=0, attn_loc='last', conv_type='Conv')
        self.Big_BLockA5 = BasicLayer1(network_depth=48, dim=96, depth=6,
                                       num_heads=4, mlp_ratio=2.,
                                       norm_layer=RLN, window_size=8,
                                       attn_ratio=0, attn_loc='last', conv_type='DWConv')
        # upsample
        self.up1 = PatchUnEmbed(patch_size=2, out_chans=48, embed_dim=96)  # x=(bs,96,64,64) -> x=(bs,48,128,128)
        self.up2 = PatchUnEmbed(patch_size=2, out_chans=24, embed_dim=48)  # x=(bs,48,128,128) -> x=(bs,24,256,256)
        # Restore
        self.patch_unembed = PatchUnEmbed(patch_size=1, out_chans=out_chans, embed_dim=24, kernel_size=3)  # x=(bs,24,256,256) 恢复
        self.zero1 = nn.Sequential(
            normalization(96),
            nn.SiLU(),
            zero_module(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1)),
        )
        self.zero2 = nn.Sequential(
            normalization(96),
            nn.SiLU(),
            zero_module(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1)),
        )

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.concat_img_prior(x)  # 输入通道数3 输出4
        x = self.patch_embed(x)  # 将图像分割为不重叠的patch 把输入图像变成通道数24 大小256
        x = self.down1(x)  # x=(bs,24,256,256) -> x=(bs,48,128,128)
        x = self.down2(x)  # x=(bs,48,128,128) -> x=(bs,96,64,64)

        skip_zero1 = self.zero1(x)
        skip_A1 = x
        x = self.Big_BLockA1(x)
        x += skip_A1
        x = skip_zero1 + x

        skip1 = x

        skip_zero2 = self.zero2(x)
        skip_A2 = x
        x = self.Big_BLockA2(x)
        x += skip_A2
        x = skip_zero2 + x

        skip2 = x
        x = self.Big_BLockA3(x)
        x = self.fusion([x, skip2]) + x  # 相加 不是concat 两个矩阵的数字直接相加
        skip_B2 = x
        x = self.Big_BLockA4(x)
        x += skip_B2
        x = self.fusion([x, skip1]) + x
        skip_B3 = x
        x = self.Big_BLockA5(x)
        x += skip_B3
        x = self.up1(x)  # x=(bs,96,64,64) -> x=(bs,48,128,128)
        x = self.up2(x)  # x=(bs,48,128,128) -> x=(bs,24,256,256)
        x = self.patch_unembed(x)  # x=(bs,24,256,256)-> x=(bs,4,256,256) 恢复
        return x

    def forward(self, x):  # x(4,3,256,256)
        H, W = x.shape[2:]  # H:256 W:256
        x = self.check_image_size(x)
        feat = self.forward_features(x)   # feat=(4,4,256,256)
        K, B = torch.split(feat, [1, 3], dim=1)  # K=(4,1,256,256) B=(4,3,256,256)
        x = K * x - B + x  # +x是最外面的残差  K(4,1,256,256)*x(4,3,256,256) - B(4,3,256,256) + x(4,3,256,256)
        x = x[:, :, :H, :W]
        return x


if __name__ == '__main__':
    net = Dehaze2().to('cuda:0')
    input = torch.rand(1, 3, 256, 256).to('cuda:0')
    output = net(input)
    print(output.shape)

