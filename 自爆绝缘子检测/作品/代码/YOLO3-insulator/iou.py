
def IOU(Reframe, GTframe):
    # 得到第一个矩形的左上坐标及宽和高
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]
    height1 = Reframe[3]

    # 得到第二个矩形的左上坐标及宽和高
    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]
    height2 = GTframe[3]
    # 计算重叠部分的宽和高
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    # 如果重叠部分为负, 即不重叠
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1.0 / (Area1 + Area2 - Area)

    return ratio

#rec_du = (left_du, top_du, width_du, height_du)  # （x, y , width, height）
#rec_s = (left_s, top_s, width_s, height_s)  # （x, y , width, height）
#iou_res = IOU(rec_du, rec_s)
# In[测试指定的文件] 
#from yolo import YOLO
#from yolo_defect import YOLO_Defect
#from PIL import Image
#import os
##import keras
#import glob
#
##import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#
##keras.backend.clear_session()
#tf.keras.backend.clear_session()
#
#
#FLAGS = {}
#
#defection = YOLO_Defect(**(FLAGS))
#path = "./dataset/test_insulator/*.JPG"
##outdir = "./result"
#valFile = {}
#
#for jpgfile in glob.glob(path):
#    name = os.path.basename(jpgfile)
#    img = Image.open(jpgfile)
##    img = cv2.imread(jpgfile)
#    print(jpgfile)
#    quexian = defection.detect_image(img)
#    print(quexian)
#    valFile[jpgfile] = quexian
#saveIOU(valFile)
# In[预测坐标数据持久化]

def saveIOU(valFile):
    k = []
    for i in valFile:
        print(i,valFile[i])
        name = os.path.basename(i)
        strr = ""
        for j in valFile[i]:
            strr = strr + str(j[1]) +","+ str(j[0]) +","+ str(j[3]-j[1]) +","+ str(j[2]-j[0]) + " "
        k.append(name+ " "+ strr+ "\n")
    
    with open("dataset/iou1.txt","w+",encoding="utf8") as f:
        f.writelines(k)
# In[测试train的文件] 
from PIL import Image
from yolo_defect import YOLO_Defect
trainPath = r"train.txt"
import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior()
#keras.backend.clear_session()
tf.keras.backend.clear_session()

FLAGS = {}
defection = YOLO_Defect(**(FLAGS))
f = open("train.txt","r+",encoding="utf-8")
lines = f.readlines()
f.close()
valFile = {}
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
    valFile[ll[0]] = point1
saveIOU(valFile)

        
# In[训练坐标持久化]
import os
f = open("train.txt","r+",encoding="utf8")
o = f.readlines()

k = []
for i in o:
    print(i)
    l = i.split(" ")
    if len(l) >= 2:
        name = os.path.basename(l[0])
        strr = ""
        for j in l[1:]:
            a = j.split(",")
            strr = strr + a[0]+","+"0"+","+str(int(a[2])-int(a[0]))+","+"128"+" "
        k.append(name + " " +strr+"\n")
    else:
      k.append(os.path.basename(i))
print(k)
f.close()
f = open("dataset/train_iou.txt","w+",encoding="utf8")
#for i in k:
#    f.write(i)
#    print(i)
f.writelines(k)
f.close()
# In[对比所有的IOU]
with open("dataset/train_iou.txt",'r+',encoding="utf8") as f:
    train = f.readlines()
    
with open("dataset/iou1.txt",'r+',encoding="utf8") as f:
    iou = f.readlines()

import math

len(iou),len(train)
sum1 = 0
iouSum = 0
for i,j in zip(iou,train):
    i1 = i.split(" ") #预测的
    j1 = j.split(" ")
    if len(i1) == 1 and len(j1) >= 1:
#        sum1 = sum1 + len(j1[1:])
        continue
    if (len(j1) == 1 and len(i1) >= 1):
#        sum1 = sum1 + len(i1[1:])
        continue
    if len(j1) == 1 and len(i1) == 1:
        continue
    min1 = min(len(i1[1:]),len(j1[1:]))
    for k in range(min1):
        
        q1 = [float(q) for q in i1[1:][k].split(",") if q != "\n"]
        
        q2 = [float(q) for q in j1[1:][k].split(",") if q != "\n"]
        
        if len(q1) == 4 and len(q2) == 4: 
            iouu = IOU(q1,q2)
            iouSum = iouSum + iouu
            print(iouu)
#        print(i1[1:][k],j1[1:][k].split(","))

#    sum1 = sum1 + max(len(i1[1:]),len(j1[1:]))
    sum1 = sum1 + min1
    

print("iou:{}".format(str(iouSum / sum1)),"iouSum:{}".format(iouSum),"sum1:{}".format(sum1))






