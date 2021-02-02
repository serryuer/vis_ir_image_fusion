# -*- coding:utf-8 -*-
# @auth hdy
# 哈尔小波变换
print('-----------import Library-------------')
import cv2
import numpy as np 
from time import *
load_image_start = time()
imgA = cv2.imread("/data/yujsh/xiaoxiannv/fusion16/dataset16/3_7.tif",-1) #载入图片A
imgB = cv2.imread("/data/yujsh/xiaoxiannv/fusion16/dataset16/3_5.tif",-1) #载入图片B
load_image_end = time()
print("load image time: ",load_image_end - load_image_start)
#heigh, wide, channel = imgA.shape #获取图像的高、宽、通道数
heigh, wide= imgA.shape
channel  =1
###############################
#临时变量、存储哈尔小波处理后的数据#
tempA1 = []                   #
tempA2 = []                   #
tempB1 = []                   #
tempB2 = []                   #
###############################
waveImgA = np.zeros((heigh, wide), np.float64) #存储A图片小波处理后数据的变量
waveImgB = np.zeros((heigh, wide), np.float64) #存储B图片小波处理后数据的变量
#水平方向的哈尔小波处理，对图片的B、G、R三个通道分别遍历进行
fusion_start_time = time()
for x in range(heigh):
    for y in range(0,wide-1,2):
        tempA1.append((np.float64(imgA[x,y]) + np.float64(imgA[x,y+1]))/2) #将图片A小波处理后的低频存储在tempA1中
        tempA2.append((np.float64(imgA[x,y]) + np.float64(imgA[x,y+1]))/2 - np.float64(imgA[x,y])) #将图片A小波处理后的高频存储在tempA2中
        tempB1.append((np.float64(imgB[x,y]) + np.float64(imgB[x,y+1]))/2) #将图片B小波处理后的低频存储在tempB1中
        tempB2.append((np.float64(imgB[x,y]) + np.float64(imgB[x,y+1]))/2 - np.float64(imgB[x,y])) #将图片B小波处理后的高频存储在tempB2中
    tempA1 = tempA1 + tempA2 #小波处理完图片A每一个水平方向数据统一保存在tempA1中
    tempB1 = tempB1 + tempB2 #小波处理完图片B每一个水平方向数据统一保存在tempB1中
    for i in range(len(tempA1)):
        waveImgA[x,i] = tempA1[i] #图片A水平方向前半段存储低频，后半段存储高频
        waveImgB[x,i] = tempB1[i] #图片B水平方向前半段存储低频，后半段存储高频
    tempA1 = [] #当前水平方向数据处理完之后，临时变量重置
    tempA2 = [] 
    tempB1 = []
    tempB2 = []
 
#垂直方向哈尔小波处理，与水平方向同理
for y in range(wide):
    for x in range(0,heigh-1,2):
        tempA1.append((np.float64(waveImgA[x,y]) + np.float64(waveImgA[x+1,y]))/2)
        tempA2.append((np.float64(waveImgA[x,y]) + np.float64(waveImgA[x+1,y]))/2 - np.float64(waveImgA[x,y]))
        tempB1.append((np.float64(waveImgB[x,y]) + np.float64(waveImgB[x+1,y]))/2)
        tempB2.append((np.float64(waveImgB[x,y]) + np.float64(waveImgB[x+1,y]))/2 - np.float64(waveImgB[x,y]))
    tempA1 = tempA1 + tempA2
    tempB1 = tempB1 + tempB2
    for i in range(len(tempA1)):
        waveImgA[i,y] = tempA1[i]
        waveImgB[i,y] = tempB1[i]
    tempA1 = []
    tempA2 = []
    tempB1 = []
    tempB2 = []
 
#求以x,y为中心的5x5矩阵的方差，  “//”在python3中表示整除，没有小数，“/”在python3中会有小数，  python2中“/”即可，“//”也行都表示整除
varImgA = np.zeros((heigh//2, wide//2), np.float64) #将图像A中低频数据求方差之后存储的变量
varImgB = np.zeros((heigh//2, wide//2), np.float64) #将图像B中低频数据求方差之后存储的变量

for x in range(heigh//2):
    for y in range(wide//2):
        #############################
        #对图片边界(或临近)的像素点进行处理
        if x - 3    <   0:
            up      =   0
        else:
            up      =   x - 3
        if x + 3    >   heigh//2:
            down    =   heigh//2
        else:
            down    =   x + 3
        if y - 3    <   0:
            left    =   0
        else:
            left    =   y - 3
        if y + 3    >   wide//2:
            right   =   wide//2
        else:
            right   =   y + 3
        #############################
        meanA, varA = cv2.meanStdDev(waveImgA[up:down,left:right]) #求图片A以x,y为中心的5x5矩阵的方差，mean表示平均值，var表示方差
        meanB, varB = cv2.meanStdDev(waveImgB[up:down,left:right]) #求图片B以x,y为中心的5x5矩阵的方差，

        varImgA[x,y] = varA #将图片A对应位置像素的方差存储在变量中
        varImgB[x,y] = varB #将图片B对应位置像素的方差存储在变量中

#求两图的权重
weightImgA = np.zeros((heigh//2, wide//2), np.float64) #图像A存储权重的变量
weightImgB = np.zeros((heigh//2, wide//2), np.float64) #图像B存储权重的变量

for x in range(heigh//2):
    for y in range(wide//2):
        weightImgA[x,y] = varImgA[x,y] / (varImgA[x,y]+varImgB[x,y]+0.00000001) #分别求得图片A与图片B的权重
        weightImgB[x,y] = varImgB[x,y] / (varImgA[x,y]+varImgB[x,y]+0.00000001) #“0.00000001”防止零除

#进行融合，高频————系数绝对值最大化，低频————局部方差准则
reImgA = np.zeros((heigh, wide), np.float64) #图像融合后的存储数据的变量
reImgB = np.zeros((heigh, wide), np.float64) #临时变量
for x in range(heigh):
    for y in range(wide):
        if x < heigh//2 and y < wide//2:
            reImgA[x,y] = weightImgA[x,y]*waveImgA[x,y] + weightImgB[x,y]*waveImgB[x,y] #对两张图片低频的地方进行权值融合数据
        else:
            reImgA[x,y] = waveImgA[x,y] if abs(waveImgA[x,y]) >= abs(waveImgB[x,y]) else waveImgB[x,y] #对两张图片高频的进行绝对值系数最大规则融合
 
#由于是先进行水平方向小波处理，因此重构是先进行垂直方向
#垂直方向进行重构
for y in range(wide):
    for x in range(heigh):
        if x%2 == 0:
            reImgB[x,y] = reImgA[x//2,y] - reImgA[x//2 + heigh//2,y] #根据哈尔小波原理，将重构后的数据存储在临时变量中
        else:
            reImgB[x,y] = reImgA[x//2,y] + reImgA[x//2 + heigh//2,y] #图片的前半段是低频后半段是高频，除以2余数为0相减，不为0相加

#水平方向进行重构，与垂直方向同理

for x in range(heigh):
    for y in range(wide):
        if y%2 ==0:
            reImgA[x,y] = reImgB[x,y//2] - reImgB[x,y//2 + wide//2]
        else:
            reImgA[x,y] = reImgB[x,y//2] + reImgB[x,y//2 + wide//2]
#			reImgA[x,y] = reImgB[x,y//2] + reImgB[x,y//2 + wide//2]
#限制图像的范围(0-255),若不限制，根据np.astype(np.uint8)的规则，会对图片产生噪声
reImgA[reImgA[:, :] < 0] = 0
reImgA[reImgA[:, :] > 65535] = 65535
fusion_end_time = time()
print("fusion time: ", fusion_end_time - fusion_start_time)
######存图像
write_start_time = time()
cv2.imwrite("/data/yujsh/xiaoxiannv/fusion16/test_result/fusion1/fusion1_3.tif", reImgA.astype(np.uint16))
write_end_time = time()
print('imwrite time: ',write_end_time - write_start_time)