import cv2,os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

image_anno_dir=r"dataset\val_anno"
image_anno_predict_dir=r"dataset\val_predict_filter_noise"

image_anno_set=os.listdir(image_anno_dir)
image_anno_predict_set=os.listdir(image_anno_predict_dir)

dice_list=[]

for image_name in image_anno_set:
    image_anno_name=os.path.join(image_anno_dir,image_name)
    image_anno_predict_name=os.path.join(image_anno_predict_dir,image_name)
    
    image_anno=cv2.imread(image_anno_name)
    image_anno_predict=cv2.imread(image_anno_predict_name)
    
    image_anno=cv2.resize(image_anno,(473,473))
    image_anno_predict=cv2.resize(image_anno_predict,(473,473))
    
    
    image_anno=image_anno[:,:,0]
    image_anno_predict=image_anno_predict[:,:,0]
    
    image_anno[image_anno>0]=1
    image_anno_predict[image_anno_predict>0]=1
    
#    image_anno=image_anno/255
#    image_anno_predict=image_anno_predict/255
    
    union = image_anno * image_anno_predict
    dice = 2*np.sum(union)/(np.sum(image_anno)+np.sum(image_anno_predict))
    dice_list.append(dice) #两个样本的dice系数
    print(image_name,':',dice)
    
dice=np.mean(dice_list) #平均dice系数 
print('平均dice系数:',dice)   
#    image_anno=cv2.imread(image_anno_name)
#    image_anno_predict=cv2.imread(image_anno_predict_name)
#    
#    image_anno=image_anno[:,:,0]
#    image_anno_predict=image_anno_predict[:,:,0]
#    
#    image_anno[image_anno>0]=1
#    image_anno_predict[image_anno_predict>0]=1
#    
#    image_anno_inds = np.where(image_anno==1)
#    image_anno_inds = np.array(image_anno_inds).T #示例掩模图绝缘子区域的坐标（位置）
#    image_anno_predict_inds = np.where(image_anno_predict==1)
#    image_anno_predict_inds = np.array(image_anno_predict_inds).T#预测淹模图绝缘子区域的坐标（位置）
#    break
##    inters_=[]
##    for l in image_anno_inds:
##        if l in image_anno_predict_inds:
##            inters_.append(l)
##     a_b=len(inters_) #两个区域的交集个数
##    a_b=len(list(set(image_anno_inds).insersection(set(image_anno_predict_inds)))) #
#    a_b=[filter 
#    a=len(image_anno_inds) #示例掩模图绝缘子区域的点的个数
#    b=len(image_anno_predict_inds) #预测淹模图绝缘子区域的点的个数
#    a_b_dice=(2*a_b)/(a+b) #两个样本的dice系数
#    dice_set.append(a_b_dice)
#    break
#dice=np.mean(dice_set) #平均dice系数


    