# -*- coding: utf-8 -*-
'''
根据main_segment.py训练好的PSPNet网络模型，对测试样本进行预测其掩模图像。
即对某个测试样本的原图，切分子块，输入到PSPNet中，对每个子块分别得到掩模，然后并组装成一个测试样本完整的掩模。
'''

import cv2,os
from matplotlib import pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
import datetime
import imutils
from skimage import measure, color
from sklearn.decomposition import PCA

from keras_segmentation.train import  find_latest_checkpoint
from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50 #,resnet50_pspnet
#from cv2 import imresize

# In[model]:
new_model = pspnet_50( n_classes=2, input_height=473, input_width=473)
file_model = 'new_model.299'
#file_model = 'new_model_val.50'
new_model.load_weights(file_model)

# In[val 测试集（不用先增强，直接拆分）]:
### 原始图像和分割的路径
image_dir =  r"dataset\val"
#
## 切分块后保存图像和分割的路径
image_block_dir =  r"dataset\block_val"
image_predict_dir =  r"dataset\val_predict"

# In[proc]:

# 指定子块的宽和高
block_width = 713
block_height = 713

if not os.path.exists(image_block_dir):
   os.makedirs(image_block_dir)
if not os.path.exists(image_predict_dir):
   os.makedirs(image_predict_dir)

#hw = []
#files_list = []
start_time=datetime.datetime.now()

image_set = os.listdir(image_dir)
icount=len(image_set)
for ii,image_name in enumerate(image_set):
   
   image = cv2.imread(os.path.join(image_dir,image_name))
   image_name = image_name[:-4] + '.png' # 改后缀名
   
   # 扩展图像，使得能够被 block_width 整除
   height_pad = block_height - np.mod(image.shape[0],block_height)
   width_pad = block_width - np.mod(image.shape[1],block_width)
   height_count = int(image.shape[0]/block_width) + int(height_pad>0)
   width_count = int(image.shape[1]/block_width)  + int(width_pad>0)
   image=np.pad(image,[(0,height_pad),(0,width_pad),(0,0)])
   
   image_compose = np.zeros_like(image)
   
   #获取每张图片的行数和列数
#   hw.append([height_count,width_count])
#   files = [] #每张图片的分块集

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

         image_block_name = image_name[:-4]+"_h"+str(hi)+"_w"+str(wi)+image_name[-4:]
         imageio.imwrite(os.path.join(image_block_dir,image_block_name), image_block)
         
         # 保存切分的子块，形成文件路径，以供PSPNet使用（也可以直接用image_block变量）
         anno_dir_path=os.path.join(image_block_dir,image_block_name)
         
         out_fname=None # 不保存预测的分块，只保存组合的整个seg
         #out_fname=os.path.join(image_predict_dir,image_block_name)
         out = new_model.predict_segmentation(inp=anno_dir_path,out_fname=out_fname)
         #print(out.shape,image_compose[hb:he,wb:we,:].shape)
         
         out=cv2.resize(out, (block_width, block_height),interpolation=cv2.INTER_NEAREST)
#         image_compose[hb:he,wb:we,0]=out*255

         image_compose[hb:he,wb:we,0]=out*255
         image_compose[hb:he,wb:we,1]=out*255
         image_compose[hb:he,wb:we,2]=out*255
         
#         image_compose=image_compose*255
   
   # 去掉pad的空白，不然掩模和原图大小不一致，在绝缘子规范化时main_predict_pca.py会有问题。
   image_compose1 = image_compose[:-height_pad,:-width_pad,:]  

   imageio.imwrite(os.path.join(image_predict_dir,image_name), image_compose1)
   duration=datetime.datetime.now()-start_time
   print('{}/{},  {} done! ------- Duration (s): {}'.format(ii,icount,image_name, duration.seconds) )
         

   
   
   
   
   
   
   