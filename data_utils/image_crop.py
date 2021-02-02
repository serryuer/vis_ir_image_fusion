from PIL import Image
import os

image_path = os.walk(r"/sdd/yujunshuai/code/image_fusion_tmp/test_result/test/vifnet_ssim_tv_endage/result/epoch_6/")
#img_path = '/sdd/yujunshuai/code/image_fusion_tmp/test_result/test/vifnet_ssim_tv_endage/result/epoch_11/fusion_0001.bmp.bmp'
save_dir = '../test_result/crop_image/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for path, dir_list, file_list in image_path:
    for file_name in file_list:
        img = Image.open(f'/sdd/yujunshuai/code/image_fusion_tmp/test_result/test/vifnet_ssim_tv_endage/result/epoch_11/{file_name}')
        crop_x = 0
        crop_y = img.size[1] - (img.size[1] - 60) / 3
        crop_w = img.size[0]
        crop_h = img.size[1]
        crop_img = img.crop((crop_x, crop_y, crop_w, crop_h))
        crop_img.save(os.path.join(save_dir, file_name))