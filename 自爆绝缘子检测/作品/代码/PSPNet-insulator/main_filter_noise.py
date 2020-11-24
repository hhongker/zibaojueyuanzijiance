import cv2,os
from matplotlib import pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
import imageio
import numpy as np
import datetime
from skimage import measure, color
from sklearn.decomposition import PCA

image_predict_dir =  r"dataset\val_predict"
image_predict_filter_noise_dir =  r"dataset\val_predict_filter_noise"

if not os.path.exists(image_predict_filter_noise_dir):
   os.makedirs(image_predict_filter_noise_dir)

image_set=os.listdir(image_predict_dir)
icount=len(image_set)


start_time=datetime.datetime.now()

#开始去掉一些噪音
for ii,image_name in enumerate(image_set):
   
#   image_name='003.png'
   image = cv2.imread(os.path.join(image_predict_dir,image_name))
   
   image2 = image[:,:,0] # 只取一个分量即可
   image2[image2>0]=1  # 原始数据的掩模，非0的有很多种数，而模型的输入要求是二值的
   
   # 采用skimage中的measure，寻找每一个连通区域
   labeled_img, num = measure.label(image2, background=0, return_num=True)
   dst = color.label2rgb(labeled_img)
   
   classes = np.unique(labeled_img)
   classes = classes[1:] # 不要背景0这一类
   for c in classes:
      inds = np.where(labeled_img==c)
      inds = np.array(inds).T
      
      # 忽略太少的点组成的连通区域
      if len(inds[:,1])<1500:
          image2[labeled_img==c]=0
      else:
          trans_pca = PCA(n_components=2).fit(inds)# 对该连通区域所有点的坐标集合，进行PCA变换
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
          [tl,tw] = np.max(inds_pca,axis=0) - np.min(inds_pca,axis=0) # pca1上的投影坐标之差的最大值作为长度tl，pca2上的坐标之差的最大值作为高度tw
          # 连通区域的长度和宽度之比，小于3，则认为是噪音
          if tl<500 or tw==0 or tl/tw<1.5:
              image2[labeled_img==c]=0
   
   image[:,:,1]=image2*255
   image[:,:,2]=image2*255
   image[:,:,0]=image2*255
#   plt.imshow(image)
#   break
   imageio.imwrite(os.path.join(image_predict_filter_noise_dir,image_name), image)
   duration=datetime.datetime.now()-start_time
   print('{}/{},  {} done! ------- Duration (s): {}'.format(ii,icount,image_name, duration.seconds) )