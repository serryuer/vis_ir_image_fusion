import logging
import cv2
import numpy as np
import os, sys
sys.path.append('/data/yujsh/xiaoxiannv/fusion')
import torch
from torch.nn import DataParallel
from torchvision import transforms
from model.VIFNet_resnet_v2 import VIFNet_resnet_v2
from PIL import Image

def test_example(model, vis_image_path, ir_image_path, save_path, device_id=-1, read_FLAG=0):
    # img_v = cv2.imread(vis_image_path,read_FLAG).astype(np.float32)
    # img_r = cv2.imread(ir_image_path,read_FLAG).astype(np.float32)
    img_v = Image.open(vis_image_path) 
    img_r = Image.open(ir_image_path)  
    if img_v.mode == 'RGB' or img_r.mode == 'RGB':
        img_v = img_v.convert('L')
        img_r = img_r.convert('L')
    loader = transforms.Compose([transforms.ToTensor()])
    img_v = loader(img_v)
    img_r = loader(img_r)
    data = torch.stack([img_v, img_r]).unsqueeze(0)
    if device_id != -1:
        data = data.cuda(device_id) 
    with torch.no_grad():
        fuse_img_tensor, _ = model(data)
        fuse_img_tensor = fuse_img_tensor.cpu().squeeze(0)
        fuse_img = transforms.ToPILImage()(fuse_img_tensor)
        fuse_img.save(save_path)

def test_examples(model, vis_image_path, ir_image_path, save_path, device_id=-1):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    vis_images = os.listdir(vis_image_path)
    ir_images = os.listdir(ir_image_path)
    assert len(vis_images) == len(ir_images), 'vis and ir image numbers not equal'
    for vis_image, ir_image in zip(vis_images, ir_images):
        assert vis_image == ir_image, 'vis and ir image name not matchs'
        test_example(model,
                    os.path.join(vis_image_path, vis_image), 
                    os.path.join(ir_image_path, ir_image),
                    os.path.join(save_dir, ir_image))

    
model_path = '/data/yujsh/xiaoxiannv/fusion/save_model/vif/best-validate-model-36.pt'

if __name__ == '__main__':
    device_id = -1
    parallel_mode = False
    model = VIFNet_resnet_v2(device_id)
    # model = DataParallel(model)
    if device_id != -1:
        model = model.cuda(device_id)
        state_dict = torch.load(model_path).state_dict()
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu')).state_dict()
        
    if parallel_mode:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.','') # remove `module.`
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()

    test_example(model, 
                '/data/yujsh/xiaoxiannv/fusion/data/vif/IR/0001.bmp',
                '/data/yujsh/xiaoxiannv/fusion/data/vif/VIS/0001.bmp',
                '/data/yujsh/xiaoxiannv/fusion/data/vif/fusion/0001.bmp')
    
    
