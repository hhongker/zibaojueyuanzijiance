
# 对输入图像进行cv2.resize到相同的大小

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:05:15 2020

@author: Administrator
"""

from keras_segmentation.train import  find_latest_checkpoint
from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K,resnet_pspnet_VOC12_v0_1
from keras_segmentation.models.pspnet import pspnet_50 #,resnet50_pspnet
from keras.utils import plot_model
from keras_segmentation.predict import get_colored_segmentation_image
import datetime,cv2
from matplotlib import pyplot as plt
import numpy as np

import os
# In[train]:

# 只有pspnet才有预训练好的数据，(ADE20k数据库：20K张图像，150类)
# 若没有模型文件，则自动下载（由于下载速度很慢，所以建议先把文件放进相应的目录）
# C:\Users\Administrator\.keras\datasets\pspnet50_ade20k.h5
pretrained_model = pspnet_50_ADE_20K()

# 在pspnet中(keras_segmentation/models/_pspnet_2.py)，
# (input_height,input_width)只有(473,473),(713,713)的pooling才有定义
new_model = pspnet_50( n_classes=2, input_height=473, input_width=473)
#new_model = pspnet_50( n_classes=2, input_height=713, input_width=713)
#new_model = pspnet_50( n_classes=2, input_height=473, input_width=713)

# 对基本Res分类网络的权重固定不训练，训练识别率会下降
#for layer in new_model.layers:
#	layerName=str(layer.name)
#	if layerName.startswith("Res_"):
#		layer.trainable=False
    
#new_model.summary()
#plot_model(new_model,to_file = 'new_model_473_713.png')  # 画出模型结构图，并保存成图片

file_model = 'new_model_org.h5'
if not os.path.exists(file_model):
   # 模型的系数很多，TitianX都要2分钟。
   # 每一层，如果权重的形状相同，则复制（最后一层由于类别不同，则不会复制）
   transfer_weights( pretrained_model , new_model  ) # transfer weights from pre-trained model to your model
   new_model.save_weights(file_model)
new_model.load_weights(file_model)


start_time=datetime.datetime.now()

# 根据有标注的训练样本进行训练
# keras_segmentationd的代码做了修改：
# model/model_utils.py:  model = Model(img_input, [o,o])
# train.py: model.compile(loss=['categorical_crossentropy',masked_categorical_crossentropy]

new_model.train(
    train_images =  "dataset/block_train",
    train_annotations = "dataset/block_train_anno/",    
    n_classes=2,
    verify_dataset=False, # 不用每次都验证图像和掩模的数据是否一致
    validate=True,
    loss_weight=1,
    val_images="dataset/block_val",
    val_annotations="dataset/block_val_anno",   
    val_steps_per_epoch=32,
    val_batch_size=2,
    checkpoints_path = "new_model" , 
    load_weights=None,#"new_model.80", # load_weights='new_model.10' 继续训练
    batch_size = 2,        # 默认2。batch_size太大的话，会导致 GPU 内存不够
    steps_per_epoch = 128, # 默认512。每一代用多少个batch, None代表自动分割，即数据集样本数/batch样本数
    epochs=140,
)
#OOM when allocating tensor with shape[2,4096,90,90] 
 
duration=datetime.datetime.now()-start_time
print('-------- Training time (s): {}'.format(duration.seconds))
 
new_model_checkpoint = find_latest_checkpoint("new_model")
new_model.load_weights(new_model_checkpoint)
out = new_model.predict_segmentation(
    inp="dataset/block_val/003_h3_w6.jpg",
    out_fname="out.png"
)

out = np.array(out).astype(np.float32) # 转换为浮点型，plt.imshow才认为图像在0到1内，即1是白的。
plt.imshow(out)

##out_color = cv2.imread("out2.png")
#out_color = get_colored_segmentation_image(out,n_classes=2)
#out_color = np.array(out_color).astype(np.int32)
#plt.imshow(out_color)

#import sys
#sys.exit()

#Epoch 60/200
#128/128  - loss: 0.0763 - softmax_loss: 0.0116 - softmax_accuracy: 0.9785 - softmax_accuracy_1: 0.9785 - val_loss: 0.7011 - val_softmax_loss: 0.1553 - val_softmax_accuracy: 0.9488 - val_softmax_accuracy_1: 0.9488
#saved  new_model.59

#Epoch 99/200
#128/128  - loss: 0.0609 - softmax_loss: 0.0095 - softmax_accuracy: 0.9828 - softmax_accuracy_1: 0.9828 - val_loss: 0.7614 - val_softmax_loss: 0.1994 - val_softmax_accuracy: 0.9624 - val_softmax_accuracy_1: 0.9624
#saved  new_model.98

#Epoch 200/200
#128/128 - loss: 0.0437 - softmax_loss: 0.0064 - softmax_accuracy: 0.9869 - softmax_accuracy_1: 0.9869 - val_loss: 1.3245 - val_softmax_loss: 0.2351 - val_softmax_accuracy: 0.9677 - val_softmax_accuracy_1: 0.9677
#saved  new_model.199
#-------- Training time (s): 15388

#200+100 （all block） new_model.99
#Epoch 50/100
#128/128  - loss: 0.0256 - softmax_loss: 0.0049 - softmax_accuracy: 0.9939 - softmax_accuracy_1: 0.9939 - val_loss: 0.1771 - val_softmax_loss: 0.0497 - val_softmax_accuracy: 0.9674 - val_softmax_accuracy_1: 0.9674
#Epoch 50/100
#128/128 - loss: 0.0174 - softmax_loss: 0.0028 - softmax_accuracy: 0.9952 - softmax_accuracy_1: 0.9952 - val_loss: 0.1486 - val_softmax_loss: 0.0274 - val_softmax_accuracy: 0.9824 - val_softmax_accuracy_1: 0.9824
#saved  new_model.99

#200+100+ loss_weight=1, 99
#Epoch 99/99
#128/128 [==============================] - 73s 570ms/step - loss: 0.0166 - softmax_loss: 0.0045 - softmax_accuracy: 0.9953 - softmax_accuracy_1: 0.9953 - val_loss: 0.0753 - val_softmax_loss: 0.0370 - val_softmax_accuracy: 0.9731 - val_softmax_accuracy_1: 0.9731
#saved  new_model.98

# In[test]:

#import cv2,os
#import numpy as np
#from matplotlib import pyplot as plt
##out = cv2.imread("out.png")
#np.unique(out[:,:])
#plt.imshow(out)
#
##seg = cv2.imread("dataset1/annotations_prepped_train/0001TP_007080.png")
##seg = cv2.imread("ADE20K/ADE_train_00014997_seg.png")
#seg = cv2.imread("dataset/block_train_anno/001_0_h4_w3.png")
#seg = np.array(seg).astype(np.float32) # 转换为浮点型，plt.imshow才认为图像在0到1内，即1是白的。
#plt.imshow(seg) 
##plt.figure(figsize=(10,10))
##plt.imshow(seg,cmap=plt.get_cmap("binary"),vmin=0,vmax=1)
#np.unique(seg[:,:,0])
#np.unique(seg[:,:,1])
#np.unique(seg[:,:,2])
#np.max(seg[:,:,0])
#np.where(seg[:,:,0]>0)
#
##pyplot.imshow(seg)
##
##images_path = "dataset/train_anno/"
##for dir_entry in os.listdir(images_path):
##   filename = os.path.join(images_path, dir_entry)
##   if os.path.isfile(filename):
##      seg = cv2.imread(filename)
##      seg[np.where(seg>0)]=1
##      cv2.imwrite(filename,seg)
#
