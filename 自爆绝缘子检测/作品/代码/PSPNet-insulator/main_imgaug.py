# -*- coding: utf-8 -*-

import cv2,os
from matplotlib import pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
import datetime

# 原始图像和分割的路径
image_dir = r"dataset\train"
image_seg_dir = r"dataset\train_anno" 

# 增强后保存图像和分割的路径
image_aug_dir = r"dataset\aug_train"
image_seg_aug_dir = r"dataset\aug_train_anno" 

## 原始图像和分割的路径
#image_dir = r"dataset\val"
#image_seg_dir = r"dataset\val_anno" 
#
## 增强后保存图像和分割的路径
#image_aug_dir = r"dataset\aug_val"
#image_seg_aug_dir = r"dataset\aug_val_anno" 

if not os.path.exists(image_aug_dir):
   os.makedirs(image_aug_dir)
if not os.path.exists(image_seg_aug_dir):
   os.makedirs(image_seg_aug_dir)

# 变换的集合，每种变换生成一个新的图像样本及其segment
# 1.先分别缩放   
transform_seqs1 = [
                 iaa.Affine(scale=0.6),   
                 iaa.Affine(scale=0.8),   
                 iaa.Affine(scale=1.2),   
                 iaa.Affine(scale=1.4),   
                 iaa.Fliplr(),           #0.镜像翻转
                 iaa.Flipud(),            #1.左右翻转
                 iaa.Affine(rotate=20),   #旋转
                 iaa.Affine(rotate=40),   
                 iaa.Affine(rotate=60),   
                 iaa.Affine(rotate=90),  
                 iaa.Affine(rotate=-20),  
                 iaa.Affine(rotate=-40),  
                 iaa.Affine(rotate=-60),     
                  # 先将图片从RGB变换到HSV,然后将H值增加10,然后再变换回RGB。
                 iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                         children=iaa.WithChannels(0, iaa.Add(10))),
                 iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                         children=iaa.WithChannels(0, iaa.Add(20))),
                 iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                         children=iaa.WithChannels(0, iaa.Add(-10))),
                 iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                         children=iaa.WithChannels(0, iaa.Add(-20))),
                 iaa.Grayscale(),                
                 ]

## 2.对每一种缩放，再先分别旋转
#transform_seqs2 = [iaa.Fliplr(),           #0.镜像翻转
#                 iaa.Flipud(),            #1.左右翻转
#                 iaa.Affine(rotate=20),   #旋转
#                 iaa.Affine(rotate=40),   
#                 iaa.Affine(rotate=60),   
#                 iaa.Affine(rotate=90),  
#                 iaa.Affine(rotate=-20),  
#                 iaa.Affine(rotate=-40),  
#                 iaa.Affine(rotate=-60), 
#                 ]

start_time=datetime.datetime.now()
image_set = os.listdir(image_dir)
for image_name in image_set:
   
   image = cv2.imread(os.path.join(image_dir,image_name))
   image_name = image_name[:-4] + '.jpg' # 改后缀名
   image_seg_name = image_name[:-4] + '.png' # 改后缀名
   if not os.path.exists(os.path.join(image_seg_dir,image_seg_name)):
      print("不存在分割文件："+image_seg_name)
      continue
   image_seg = cv2.imread(os.path.join(image_seg_dir,image_seg_name))

   # 把掩模的数据变为只有0和1两类
   image_seg[np.where(image_seg>0)]=1

#   # 图片太大，先缩小到原来的1/4进行测试   
#   height,width = int(image.shape[0]/4),int(image.shape[1]/4)
#   image = cv2.resize(image, (width, height))
#   image_seg = cv2.resize(image_seg, (width, height),interpolation=cv2.INTER_NEAREST)
   
   # 把原始图像也复制进去
   imageio.imwrite(os.path.join(image_aug_dir,image_name), image)
   imageio.imwrite(os.path.join(image_seg_aug_dir,image_seg_name), image_seg)
   
   #将标签转换为SegmentationMapOnImage类型，施加相同变换后能得到正确的标签（不会线性插值）
   image_seg = ia.SegmentationMapsOnImage(image_seg, shape=image.shape)
   
   ind=0
   for ind1,tr1 in enumerate(transform_seqs1):
#      if ind1<13:
#         continue
      
      image_aug1 = tr1.augment_image(image) 
      image_aug_seg1 = tr1.augment_segmentation_maps(image_seg).get_arr().astype(np.uint8) 
      
      ind=ind+1
      image_aug_name = image_name[:-4]+"_"+str(ind1)+image_name[-4:]
      image_seg_aug_name = image_seg_name[:-4]+"_"+str(ind1)+image_seg_name[-4:]
      imageio.imwrite(os.path.join(image_aug_dir,image_aug_name), image_aug1)
      imageio.imwrite(os.path.join(image_seg_aug_dir,image_seg_aug_name), image_aug_seg1)
      
      #image_aug_seg1 = ia.SegmentationMapsOnImage(image_aug_seg1, shape=image.shape)
#      for ind2,tr2 in enumerate(transform_seqs2):
#         image_aug = tr2.augment_image(image_aug1) 
#         image_aug_seg = tr2.augment_segmentation_maps(image_aug_seg1).get_arr().astype(np.uint8) 
#         
#         ind=ind+1
#         image_aug_name = image_name[:-4]+"_"+str(ind1)+"_"+str(ind2)+'.png' #image_name[-4:]
#         imageio.imwrite(os.path.join(image_aug_dir,image_aug_name), image_aug)
#         imageio.imwrite(os.path.join(image_seg_aug_dir,image_aug_name), image_aug_seg)
         
         #plt.imshow(image_aug)
         #plt.imshow(image_aug_seg)

   duration=datetime.datetime.now()-start_time
   print('{} done! ------- Duration (s): {}'.format(image_name, duration.seconds) )
#   break
