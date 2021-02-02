
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
from PIL import Image
from torchvision import transforms
#创建30个文件夹
def mkdir(path):  # 判断是否存在指定文件夹，不存在则创建
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:

        return False


def preprocess_image(img_name):
    img_v = Image.open(f'../open_source/Pytorch_VIF-Net-master/data/TNO/{img_name}/1.bmp')
    img_r = Image.open(f'../open_source/Pytorch_VIF-Net-master/data/TNO/{img_name}/2.bmp')
    if img_v.mode == 'RGB' or img_r.mode == 'RGB':
        img_v = img_v.convert('L')
        img_r = img_r.convert('L')
    img_v_tensor = transforms.ToTensor()(img_v)
    img_r_tensor = transforms.ToTensor()(img_r)
    input = torch.stack([img_v_tensor, img_r_tensor]).unsqueeze(0)
    return input


class FeatureVisualization():
    def __init__(self,input,selected_layer,model_path):
        self.input=input
        self.selected_layer=selected_layer
        self.model_path = model_path
        #self.pretrained_model = models.vgg16(pretrained=True).features
        #print( self.pretrained_model)


    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.input
        print("input shape",input.shape)
        x=input
        for index,layer in enumerate(self.pretrained_model):
            #print(index)
            #print(layer)
            x=layer(x)
            if (index == self.selected_layer):
                return x

    def get_single_feature(self):
        features=self.get_feature()
        print("features.shape",features.shape)
        feature=features[:,0,:,:]
        print(feature.shape)
        feature=feature.view(feature.shape[1],feature.shape[2])
        print(feature.shape)
        return features

    def save_feature_to_img(self):
        #to numpy
        features=self.get_single_feature()
        for i in range(features.shape[1]):
            feature = features[:, i, :, :]
            feature = feature.view(feature.shape[1], feature.shape[2])
            feature = feature.data.numpy()
            # use sigmod to [0,1]
            feature = 1.0 / (1 + np.exp(-1 * feature))
            # to [0,255]
            feature = np.round(feature * 255)
            print(feature[0])
            mkdir('./feature/' + str(self.selected_layer))
            feature.save('./feature/'+ str( self.selected_layer)+'/' +str(i)+'.jpg', feature)
if __name__=='__main__':
    # get class
    for  k in range(30):
        myClass=FeatureVisualization('/home/lqy/examples/TRP.PNG',k)
        print (myClass.pretrained_model)
        myClass.save_feature_to_img()


