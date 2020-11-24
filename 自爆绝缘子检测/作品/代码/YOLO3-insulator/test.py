# In[预测指定的图片]
#from yolo import YOLO
from yolo_defect import YOLO_Defect
from PIL import Image
import os
#import keras
import glob

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#keras.backend.clear_session()
tf.keras.backend.clear_session()


FLAGS = {}

defection = YOLO_Defect(**(FLAGS))
path = "./dataset/test_insulator/*.JPG"
#outdir = "./result"
valFile = {}

for jpgfile in glob.glob(path):
    name = os.path.basename(jpgfile)
    img = Image.open(jpgfile)
#    img = cv2.imread(jpgfile)
    print(jpgfile)
    quexian = defection.detect_image(img)
    print(quexian)
    valFile[jpgfile] = quexian
    

    
# In[画出图像]

for valF in valFile:
    print(valF,valFile[valF])
    drawbbox(valFile[valF],valF)
   
    
#drawbbox([[0,1470,128,1597]],"test_insulator/001_2_0_2.jpg")
    
# In[画图的函数]
import cv2
import os 
#import matplotlib.pyplot as plt

def drawbbox(points,ImgPath=r"test_insulator/", savePath=r"dataset/saveVal"):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
    im = cv2.imread(ImgPath)
    
#    xbb = 128 / 100
#    ybb = 2048 / 2000
    
#        cv2.polylines(im, [points], True, color=(0, 0, 255),thickness=2)

    for point in points:
#    if len(points):
        print( (point[1],point[0]), (point[3],point[2]) )
#        cv2.rectangle(im, (point[1],point[0]), (point[3],point[2]), (0, 0, 255),2)
        cv2.rectangle(im, (point[1],0), (point[3],128), (0, 0, 255),2)
        
    print(os.path.join(savePath, os.path.basename(ImgPath)))
    cv2.imwrite(os.path.join(savePath,os.path.basename(ImgPath)), im)

    
#    if imm:
#        cv2.imwrite(os.path.join(savePath, os.path.basename(ImgPath)), imm)

# In[预测train的图片]
#import numpy as np
from PIL import Image
from yolo_defect import YOLO_Defect
trainPath = r"train.txt"
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#keras.backend.clear_session()
tf.keras.backend.clear_session()

FLAGS = {}
defection = YOLO_Defect(**(FLAGS))
f = open("train.txt","r+",encoding="utf-8")
lines = f.readlines()
f.close()
for i in lines:
#    print(i)
    ll = i.split(" ")
    point2 = 0;
#    print(ll[0])
    if ll[0].endswith("\n"):
        ll[0] = ll[0][:-1]
#        print(ll[0])
    img = Image.open(ll[0])
    point1 = defection.detect_image(img)
    if len(ll)>=2:
        point2 = [[int(o) for o in j.split(",")[:-1]] for j in ll[1:]]
    drawbbox2(point1,point2,ll[0]) #point2是手工测的点，蓝色
#    print(point2)

    
# In[]
import cv2
import os 
#import matplotlib.pyplot as plt

def drawbbox2(points1,points2,ImgPath=r"test_insulator/", savePath=r"dataset/saveVal"):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
    im = cv2.imread(ImgPath)
    
#    xbb = 128 / 100
#    ybb = 2048 / 2000
    
#        cv2.polylines(im, [points], True, color=(0, 0, 255),thickness=2)

    for point in points1:
#    if len(points):
        print( (point[1],point[0]), (point[3],point[2]) )
#        cv2.rectangle(im, (point[1],point[0]), (point[3],point[2]), (0, 0, 255),2)
        cv2.rectangle(im, (point[1],0), (point[3],128), (0, 0,255),2)
       
    if point2:    
        for point in points2:
    #    if len(points):
#            print( (point[1],point[0]), (point[3],point[2]) )
    #        cv2.rectangle(im, (point[1],point[0]), (point[3],point[2]), (0, 0, 255),2)
            cv2.rectangle(im, (point[0],0), (point[2],128), (255, 0, 0),2)
        
    print(os.path.join(savePath, os.path.basename(ImgPath)))
    cv2.imwrite(os.path.join(savePath,os.path.basename(ImgPath)), im)








