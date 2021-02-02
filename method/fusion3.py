# import cv2
# import numpy as np
# A = cv2.imread('/data/yujsh/xiaoxiannv/fusion/dataset16/0_7.tif',-1).astype(np.float64)
# B = cv2.imread('/data/yujsh/xiaoxiannv/fusion/dataset16/0_5.tif',-1).astype(np.float64)
# # generate Gaussian pyramid for A
# G = A.copy()
# gpA = [G]
# for i in np.arange(6):     #将苹果进行高斯金字塔处理，总共六级处理
#     G = cv2.pyrDown(G)
#     gpA.append(G)
# # generate Gaussian pyramid for B
# G = B.copy()
# gpB = [G]
# for i in np.arange(6):  # #将橘子进行高斯金字塔处理，总共六级处理
#     G = cv2.pyrDown(G)
#     gpB.append(G)
# # generate Laplacian Pyramid for A
# lpA = [gpA[5]]               
# for i in np.arange(5,0,-1):    #将苹果进行拉普拉斯金字塔处理，总共5级处理
#     GE = cv2.pyrUp(gpA[i])
#     L = cv2.subtract(gpA[i-1],GE)
#     lpA.append(L)
# # generate Laplacian Pyramid for B
# lpB = [gpB[5]]
# for i in np.arange(5,0,-1):    #将橘子进行拉普拉斯金字塔处理，总共5级处理
#     GE = cv2.pyrUp(gpB[i])
#     L = cv2.subtract(gpB[i-1],GE)
#     lpB.append(L)
# # Now add left and right halves of images in each level
# #numpy.hstack(tup)
# #Take a sequence of arrays and stack them horizontally
# #to make a single array.
# LS = []
# for la,lb in zip(lpA,lpB):
#     rows,cols,dpt = la.shape
#     ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))    #将两个图像的矩阵的左半部分和右半部分拼接到一起
#     LS.append(ls)
# # now reconstruct
# ls_ = LS[0]   #这里LS[0]为高斯金字塔的最小图片
# for i in xrange(1,6):                        #第一次循环的图像为高斯金字塔的最小图片，依次通过拉普拉斯金字塔恢复到大图像
#     ls_ = cv2.pyrUp(ls_)
#     ls_ = cv2.add(ls_, LS[i])                #采用金字塔拼接方法的图像
# # image with direct connecting each half
# real = np.hstack((A[:,:cols/2],B[:,cols/2:]))   #直接的拼接
# cv2.imwrite('/data/yujsh/xiaoxiannv/fusion/test_result/fusion/3_0_p.tif',ls_rec.astype(np.uint16))
# cv2.imwrite('/data/yujsh/xiaoxiannv/fusion/test_result/fusion/3_0_r.tif',real.astype(np.uint16))


import cv2
import numpy as np,sys

A = cv2.imread('/data/yujsh/xiaoxiannv/fusion/dataset16/0_7.tif',-1).astype(np.float64)
B = cv2.imread('/data/yujsh/xiaoxiannv/fusion/dataset16/0_5.tif',-1).astype(np.float64)

# generate Gaussian pyramid for A
G = A.copy()
print(G)
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

lpA = [gpA[5]]
for i in range(6,0,-1):
    print(i)
    GE = cv2.pyrUp(gpA[i])
    GE=cv2.resize(GE,gpA[i - 1].shape[-2::-1])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B

lpB = [gpB[5]]
for i in range(6,0,-1):
    print(i)
    GE = cv2.pyrUp(gpB[i])
    GE = cv2.resize(GE, gpB[i - 1].shape[-2::-1])
    L = cv2.subtract(gpB[i-1],GE)
    print(L.shape)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
lpAc=[]
for i in range(len(lpA)):
    b=cv2.resize(lpA[i],lpB[i].shape[-2::-1])
    lpAc.append(b)
print(len(lpAc))
print(len(lpB))
j=0
for i in zip(lpAc,lpB):
    la,lb = i
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    j=j+1
    LS.append(ls)

ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_= cv2.resize(ls_, LS[i].shape[-2::-1])
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
B= cv2.resize(B, A.shape[-2::-1])
real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
cv2.imwrite('/data/yujsh/xiaoxiannv/fusion/test_result/fusion/3_0_p.tif',ls_.astype(np.uint16))
cv2.imwrite('/data/yujsh/xiaoxiannv/fusion/test_result/fusion/3_0_r.tif',real.astype(np.uint16))