# -*- coding: utf-8 -*-
'''
基于主成分的旋转和裁剪的规范化绝缘子区域算法。

用PSPNet提取了绝缘子所在区域的二值掩模之后，
找出每个连通区域（1个连通区域可能包含1个或2个并排的绝缘子串）的坐标集合，
求连通区域的坐标的均值点，和坐标点集合的PCA的两个方向，求坐标集合在分别在两个方向上的投影。
得到投影的最小值和最大值的差，即可得到连通区域的长度和宽度（若长度和宽度之比小于3，则认为是噪音，忽略）。
把绝缘子串旋转到水平方向，规范化为相同的尺寸128×2048。
把每串绝缘子单独切分成一个独立的图像。

得到了每串绝缘子规范化的图片之后，还要标记自爆点的坐标（若存在的话）
并形成train.txt文件，以供YOLO模型使用。
'''

import cv2,os,imutils
from matplotlib import pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
import datetime

from skimage import measure, color
from sklearn.decomposition import PCA

# In[data]:
### 原始图像的路径
image_dir =  r"dataset\val"
#image_dir =  r"dataset\train"
#
## 相应掩模的路径
#image_predict_dir =  r"dataset\train_anno"
#image_predict_dir =  r"dataset\val_anno"   # 训练时，可以直解使用已知的掩模
image_predict_dir =  r"dataset\val_predict" # 测试时，用模型预测得到的掩模
#image_predict_dir =  r"dataset\train_anno_proc" # 把相交的绝缘子先切开

# 绝缘子区域的长方形子块保存的路径（若没有该目录，则自动创建）
image_predict_pca_dir = r"dataset\val_predict_pca"

# In[proc]:

if not os.path.exists(image_predict_pca_dir):
   os.makedirs(image_predict_pca_dir)

start_time=datetime.datetime.now()

image_set = os.listdir(image_dir)
icount=len(image_set)

for ii,image_name in enumerate(image_set):
   
#   image_name='003.jpg'
   
   image = cv2.imread(os.path.join(image_dir,image_name))
   image_name = image_name[:-4] + '.jpg' # 改后缀名为小写
   image_seg_name = image_name[:-4] + '.png' # 掩模的后缀名为png，不要用jpg压缩
   if not os.path.exists(os.path.join(image_predict_dir,image_seg_name)):
      print("不存在分割文件："+image_seg_name)
#      continue
   
   image_seg = cv2.imread(os.path.join(image_predict_dir,image_seg_name))
   image_seg = image_seg[:,:,0] # 只取一个分量即可
   image_seg[image_seg>0]=1  # 原始数据的掩模，非0的有很多种数，而模型的输入要求是二值的
   
#   # 测试：原图缩小一半
#   height,width = int(image.shape[0]/2), int(image.shape[1]/2) 
#   image=cv2.resize(image, (width, height))
#   image_seg=cv2.resize(image_seg, (width, height),interpolation=cv2.INTER_NEAREST)

   # 采用skimage中的measure，寻找每一个连通区域
   labeled_img, num = measure.label(image_seg, background=0, return_num=True)
   dst = color.label2rgb(labeled_img)
#   plt.figure(dpi=150)
#   plt.imshow(dst)
#   imageio.imwrite('dst.jpg', dst)
#   break
   
   classes = np.unique(labeled_img)
   classes = classes[1:] # 不要背景0这一类
   for c in classes:
      inds = np.where(labeled_img==c)
      inds = np.array(inds).T
      
      # 忽略太少的点组成的连通区域
      if len(inds[:,1])<100:
         continue
      
#      imagec = np.zeros_like(image_seg)
#      imagec[inds[:,0],inds[:,1]] = 255;
#      plt.figure(dpi=100)
#      plt.imshow(imagec)
      
      # 对该连通区域所有点的坐标集合，进行PCA变换
      trans_pca = PCA(n_components=2).fit(inds)
      pcas = trans_pca.components_  # PCA的两个主成分，即为点的坐标集合的主要方向和次要方向
      
      # 最主要的方向，计算方向角度theta，准备旋转theta角变成水平方向
      pca1 = pcas[0] 
      sinp = pca1[0]/np.linalg.norm(pca1)
      cosp = pca1[1]/np.linalg.norm(pca1)
      theta = abs(np.arcsin(sinp))  # [0, pi/2]
      if pca1[0]*pca1[1]>0: 
         theta=-theta
      theta = theta*180/np.pi #将弧度制转为角度制
      
      inds_pca = trans_pca.transform(inds)
      
#      plt.figure()
#      plt.scatter(inds_pca[:,0],inds_pca[:,1],marker='.')

      # pca1上的投影坐标之差的最大值作为长度tl，pca2上的坐标之差的最大值作为高度tw
      [tl,tw] = np.max(inds_pca,axis=0) - np.min(inds_pca,axis=0)

#      lower_q=np.quantile(inds_pca[:,1],0.05) # 下分位数，2%的位置，去除噪音
#      uper_q =np.quantile(inds_pca[:,1],0.95) # 上分位数
#      tw=uper_q-lower_q
#      tw=tw*1.3
      
      # 连通区域的长度和宽度之比，小于3，则认为是噪音
      if tl<100 or tw==0 :  #or tl/tw<3
         continue

      # 统计直方图，如果最大bin的点的数量显著大于中间bin的点的数量，则需要分为两串绝缘子
      #plt.figure()
      counts,binc,aa=plt.hist(inds_pca[:,1],21)
      
      #break
   
      if np.max(counts) - counts[10] > np.max(counts)/20:
         incount = 2
         tw = tw/2
         i1 = np.where(inds_pca[:,1]>0)[0]   # 第1串绝缘子的点的下标
         i2 = np.where(inds_pca[:,1]<=0)[0]  # 第2串绝缘子的点的下标
         
         inds1=inds[i1,:]
         inds2=inds[i2,:]
      else:
         incount = 1
         inds1=inds
         inds2=inds

      tw = int(tw) # 绝缘子串的宽度，取整
      for ini in range(incount):
         if ini==0: inds=inds1 
         else: inds=inds2
         
         inds_mean = np.mean(inds,axis=0) # 质心
         maxs = np.max(inds,axis=0)
         mins = np.min(inds,axis=0)
         
         # 在原始图像上，裁剪出一个绝缘子所在的矩形区域
         image1 = image[mins[0]:maxs[0],mins[1]:maxs[1],:]
#         plt.figure(dpi=200)
#         plt.imshow(image1)
       
         # 把矩形区域进行旋转theta角度，使得绝缘子水平放置
         image1_rotated = imutils.rotate_bound(image1, theta)
#         plt.figure(dpi=200)
#         plt.imshow(image1_rotated)
         
         # 继续把上下多余的部分裁剪掉
         hh,ww = image1_rotated.shape[0],image1_rotated.shape[1]
         
         aa = int((hh-tw)/2)
         if aa>0: 
            image1_rotated = image1_rotated[aa:-aa,:,:]

         # 高度缩放到128         
         width = int(128*image1_rotated.shape[1]/image1_rotated.shape[0])
         image1_rotated1=cv2.resize(image1_rotated, (width,128))
         
         # 宽度缩放到2048或者两边补0
         if width > 2048: 
            width=2048
            image_box=cv2.resize(image1_rotated, (2048,128))
         else:
            pad = int((2048-width)/2)
            image_box = np.pad(image1_rotated1,[(0,0),(pad,pad),(0,0)])
         
         image_box_name = image_name[:-4]+"_"+str(c)+'_'+str(ini)+image_name[-4:]
         imageio.imwrite(os.path.join(image_predict_pca_dir,image_box_name), image_box)
         
#         break
#      break
         
   duration=datetime.datetime.now()-start_time
   print('{}/{},  {} done! ------- Duration (s): {}'.format(ii,icount,image_name, duration.seconds) )
         
#   break


   
   
   
   
   
   
   