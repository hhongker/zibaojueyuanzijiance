# -*- coding: utf-8 -*-
'''
第1步：划分训练和验证样本，并保存到不同目录。
第2步：对训练样本增强，并保存到不同目录。
第3步：对增强后的训练样本划分小块，713*713像素为一块，不足的补0。掩模对应切割。对验证样本也切块，保存到不同目录。
第4步：用训练样本的子块训练，用验证样本的子块验证。
第5步：对某个测试样本的原图，切分子块，对每个子块分别得到掩模，然后并组装成一个测试样本完整的掩模，计算Dice系数。
'''

import cv2,os
from matplotlib import pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
import datetime

# In[train]:
## 原始图像和分割的路径
#image_dir = r"dataset\aug_train"
#image_seg_dir = r"dataset\aug_train_anno" 
#
## 切分块后保存图像和分割的路径
#image_block_dir = r"dataset\block_train"
#image_seg_block_dir = r"dataset\block_train_anno" 

# In[val 测试集（不用先增强，直接拆分）]:
## 原始图像和分割的路径
image_dir =  r"dataset\val"
image_seg_dir = r"dataset\val_anno" 
#
## 切分块后保存图像和分割的路径
image_block_dir =  r"dataset\block_val"
image_seg_block_dir =r"dataset\block_val_anno" 

# In[proc]:

# 指定子块的宽和高
block_width = 713
block_height = 713

if not os.path.exists(image_block_dir):
   os.makedirs(image_block_dir)
if not os.path.exists(image_seg_block_dir):
   os.makedirs(image_seg_block_dir)

start_time=datetime.datetime.now()

image_set = os.listdir(image_dir)
icount=len(image_set)
for ii,image_name in enumerate(image_set):
   
   image = cv2.imread(os.path.join(image_dir,image_name))
   image_name = image_name[:-4] + '.jpg' # 改后缀名
   image_seg_name = image_name[:-4] + '.png' # 掩模图像的后缀名，使得不会被压缩
   if not os.path.exists(os.path.join(image_seg_dir,image_seg_name)):
      print(os.path.join(image_seg_dir,image_seg_name))
      print("不存在分割文件："+image_seg_name)
      continue
   image_seg = cv2.imread(os.path.join(image_seg_dir,image_seg_name))

   # 扩展图像，使得能够被 block_width 整除
   height_pad = block_height - np.mod(image.shape[0],block_height)
   width_pad = block_width - np.mod(image.shape[1],block_width)
   height_count = int(image.shape[0]/block_width) + int(height_pad>0)
   width_count = int(image.shape[1]/block_width)  + int(width_pad>0)
   image=np.pad(image,[(0,height_pad),(0,width_pad),(0,0)])
   image_seg=np.pad(image_seg,[(0,height_pad),(0,width_pad),(0,0)])
   
   # 把掩模的数据变为只有0和1两类
   image_seg[np.where(image_seg>0)]=1
   
   #block_height = int(image.shape[0]/height_count)
   #block_width = int(image.shape[1]/width_count)
   for hi in range(height_count):
      hb = hi*block_height # 子块开始的高度下标
      if hi==height_count-1:
         he = image.shape[0]
      else:
         he = hb+block_height
      for wi in range(width_count):
         wb = wi*block_width # 子块开始的宽度下标
         if wi==width_count-1:
            we = image.shape[1]
         else:
            we = wb+block_width
            
         image_block = image[hb:he,wb:we,:]
         image_block_seg = image_seg[hb:he,wb:we,:]
         
#         # 如果掩模没有绝缘子，则不需要保存该子快
#         bg = np.where(image_block_seg>0)
#         if len(bg[0])==0:
#            continue
   
         image_block_name = image_name[:-4]+"_h"+str(hi)+"_w"+str(wi)+image_name[-4:]
         image_block_seg_name = image_seg_name[:-4]+"_h"+str(hi)+"_w"+str(wi)+image_seg_name[-4:]
         imageio.imwrite(os.path.join(image_block_dir,image_block_name), image_block)
         imageio.imwrite(os.path.join(image_seg_block_dir,image_block_seg_name), image_block_seg)
         
         #plt.imshow(image_block)
         #plt.imshow(image_block_seg)

   duration=datetime.datetime.now()-start_time
   print('{}/{},  {} done! ------- Duration (s): {}'.format(ii,icount,image_name, duration.seconds) )
#   break
