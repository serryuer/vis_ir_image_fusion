import cv2
import sys, os
from tqdm import tqdm

source_vis_image_path = '/data/yujsh/xiaoxiannv/image_vis.tif'
source_ir_image_path = '/data/yujsh/xiaoxiannv/image_ir.tif'
crop_w, crop_h = int(sys.argv[1]), int(sys.argv[1]) 
stride = int(sys.argv[2])

save_dir = f'/data/yujsh/xiaoxiannv/data/{crop_h}_{stride}'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir, "IV"))
    os.mkdir(os.path.join(save_dir, "IR"))

img_v = cv2.imread(source_vis_image_path, -1)
img_r = cv2.imread(source_ir_image_path, -1)
width, height = img_r.shape

start_x = 0
count = 0

while start_x + crop_w < width:
    start_y = 0
    while start_y + crop_h < height:
        valid = True
        repeat = 0
        for i in range(crop_w):
            for j in range(crop_h):
                if img_r[start_x+i, start_y+j] == 0 or img_v[start_x+i, start_y+j] == 0:
                    repeat += 1
                else:
                    repeat = 0
                if repeat == 10:
                    valid = False
                    break
        if valid:
            cv2.imwrite(f'{save_dir}/IR/{count}.tif',img_r[start_x:start_x+crop_w, start_y:start_y+crop_h])
            cv2.imwrite(f'{save_dir}/IV/{count}.tif',img_v[start_x:start_x+crop_w, start_y:start_y+crop_h])
            count += 1
            print(f'{count} : {start_x}, {start_y} succ')
        else:
            print(f'{count} : {start_x}, {start_y} fail')
        start_y += stride
    start_x += stride

