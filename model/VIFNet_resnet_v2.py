from model.loss import *
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os, sys
sys.path.append('/data/yujsh/xiaoxiannv/fusion')
from data_utils.TNODataset16 import TNODataset

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()# 负数部分的参数会变
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual

class FeatureExtractionBlock(nn.Module):
    def __init__(self):
        super(FeatureExtractionBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3,stride = 1, padding=1),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)

    def forward(self, x):
        block1 = self.block1(x)

        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        # if not model.training:
        #     save_dir = '../encoder_feature_visual/VIFNet_resnet_v2/'
        #     if not os.path.exists(save_dir):
        #         os.mkdir(save_dir)
        #     feature_c1 = block1.cpu().squeeze(0)
        #     feature_c1 = transforms.ToPILImage()(feature_c1)
        #     feature_c1.save(os.path.join(save_dir, f'{feature_c1}.bmp'))
        #     feature_c2 = block2.cpu().squeeze(0)
        #     feature_c2 = transforms.ToPILImage()(feature_c2)
        #     feature_c2.save(os.path.join(save_dir, f'{feature_c2}.bmp'))
        #     feature_c3 = block3.cpu().squeeze(0)
        #     feature_c3 = transforms.ToPILImage()(feature_c3)
        #     feature_c3.save(os.path.join(save_dir, f'{feature_c3}.bmp'))
        #     feature_c4 = block4.cpu().squeeze(0)
        #     feature_c4 = transforms.ToPILImage()(feature_c4)
        #     feature_c4.save(os.path.join(save_dir, f'{feature_c4}.bmp'))
        #     feature_c5 = block1.cpu().squeeze(0)
        #     feature_c5 = transforms.ToPILImage()(feature_c5)
        #     feature_c5.save(os.path.join(save_dir, f'{feature_c5}.bmp'))
        return block5


class FeatureExtractionBlock2(nn.Module):
    def __init__(self):
        super(FeatureExtractionBlock2, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3,stride = 1, padding=1),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)

    def forward(self, x):
        block1 = self.block1(x)

        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        # if not model.training:
        #     save_dir = '../encoder_feature_visual/VIFNet_resnet_v2/'
        #     if not os.path.exists(save_dir):
        #         os.mkdir(save_dir)
        #     feature_c1 = block1.cpu().squeeze(0)
        #     feature_c1 = transforms.ToPILImage()(feature_c1)
        #     feature_c1.save(os.path.join(save_dir, f'{feature_c1}.bmp'))
        #     feature_c2 = block2.cpu().squeeze(0)
        #     feature_c2 = transforms.ToPILImage()(feature_c2)
        #     feature_c2.save(os.path.join(save_dir, f'{feature_c2}.bmp'))
        #     feature_c3 = block3.cpu().squeeze(0)
        #     feature_c3 = transforms.ToPILImage()(feature_c3)
        #     feature_c3.save(os.path.join(save_dir, f'{feature_c3}.bmp'))
        #     feature_c4 = block4.cpu().squeeze(0)
        #     feature_c4 = transforms.ToPILImage()(feature_c4)
        #     feature_c4.save(os.path.join(save_dir, f'{feature_c4}.bmp'))
        #     feature_c5 = block1.cpu().squeeze(0)
        #     feature_c5 = transforms.ToPILImage()(feature_c5)
        #     feature_c5.save(os.path.join(save_dir, f'{feature_c5}.bmp'))
        return block5
class FeatureFusionBlock(nn.Module):
    def __init__(self):
        super(FeatureFusionBlock, self).__init__()

    def forward(self, IA, IB):
        out = torch.cat((IA, IB), 1)
        return out


class BNFusionImageReconstructionBlock(nn.Module):
    def __init__(self):
        super(BNFusionImageReconstructionBlock, self).__init__()
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU())

    def forward(self, x):
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.C6(x)
        return x


class VIFNet_resnet_v2(nn.Module):
    def __init__(self, device_id):
        super(VIFNet_resnet_v2, self).__init__()
        self.Feature_extraction_block = FeatureExtractionBlock()
        self.Feature_extraction_block2 = FeatureExtractionBlock2()
        self.Feature_fusion_block = FeatureFusionBlock()
        self.Feature_reconstruction_block = BNFusionImageReconstructionBlock()

        self.loss = SSIM_Loss(device_id)

    def forward(self, data):
        IA = data[:, 0]
        IB = data[:, 1]
        IA_features = self.Feature_extraction_block(IA)
        IB_features = self.Feature_extraction_block2(IB)
        x = self.Feature_fusion_block(IA_features, IB_features)
        x = self.Feature_reconstruction_block(x)
        return x, self.loss(IA, IB, x)

if __name__ == '__main__':
    dataset = TNODataset('../datasets/preprocessed_1channel_TNO/')
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    print(f'dataset length : {len(dataset)}')
    model = VIFNet_resnet_v2().cuda()
    for batch_count, batch_data in enumerate(train_loader):
        batch_data = batch_data.cuda()
        output = model(batch_data)
        print(output)

