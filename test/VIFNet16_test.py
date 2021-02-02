import logging

import os, sys
sys.path.append('/data/yujsh/xiaoxiannv/fusion16')
import torch
from torch.nn import DataParallel
from torchvision import transforms
from model.VIFNet_resnet_v2 import VIFNet_resnet_v2
    
test_dir = int(sys.argv[1])

if __name__ == '__main__':
    use_cuda = False
    parallel_mode = True
    model = VIFNet_resnet_v2(use_cuda)
    # model = DataParallel(model)
    if use_cuda:
        model = model.cuda()
        state_dict = torch.load('/data/yujsh/xiaoxiannv/fusion16/save_model/best-validate-model.pt').state_dict()
    else:
        state_dict = torch.load('/data/yujsh/xiaoxiannv/fusion16/save_model/best-validate-model.pt', map_location=torch.device('cpu')).state_dict()
        
    if parallel_mode:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.','') # remove `module.`
            new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    
    def merge_vis_ir(vis_image_path, ir_image_path, save_path):
        #img_v = Image.open(vis_image_path)
        #img_r = Image.open(ir_image_path)
        img_v = cv2.imread(vis_image_path,-1).astype(np.float32)
        img_r = cv2.imread(ir_image_path,-1).astype(np.float32)
        # if img_v.mode == 'RGB' or img_r.mode == 'RGB':
        #     img_v = img_v.convert('L')
        #     img_r = img_r.convert('L')
        img_v_tensor = transforms.ToTensor()(img_v)
        img_r_tensor = transforms.ToTensor()(img_r)
        input = torch.stack([img_v_tensor, img_r_tensor]).unsqueeze(0)
        with torch.no_grad():
            fuse_img_tensor, _ = model(input)
            fuse_img_tensor = fuse_img_tensor.cpu().squeeze(0)
            fuse_img = transforms.ToPILImage()(fuse_img_tensor)
            fuse_img.save(save_path)
    
    if test_dir == 1:
        vis_image_path = sys.argv[2]
        ir_image_path = sys.argv[3]
        save_dir = sys.argv[4]
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        vis_images = os.listdir(vis_image_path)
        ir_images = os.listdir(ir_image_path)
        assert len(vis_images) == len(ir_images), 'vis and ir image numbers not equal'
        for vis_image, ir_image in zip(vis_images, ir_images):
            assert vis_image == ir_image, 'vis and ir image name not matchs'
            merge_vis_ir(os.path.join(vis_image_path, vis_image), 
                         os.path.join(ir_image_path, ir_image),
                         os.path.join(save_dir, ir_image))
    else:
        merge_vis_ir(sys.argv[2], sys.argv[3], sys.argv[4])
