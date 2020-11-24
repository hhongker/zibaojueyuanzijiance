# coding: utf-8
import os
import random
import shutil	
def moveFile(data_base):
    
    
	
    fileDir = data_base+"init/val/"  # 原文件夹路径
    fileNooaDir = data_base+"init/val_anno/"
    xml_dir= data_base+'init/xml/'

    train_dir = data_base+'train/'  # 移动到新的文件夹路径
    test_dir = data_base+'val/'
    
    train_nooa_dir = data_base+'train_anno/'
    test_nooa_dir = data_base+'val_anno/'

    train_xml_dir = data_base+'train_xml/'
    test_xml_dir= data_base+'val_xml/'


    #后缀名.jpg更改为.png
 #   image_set_old = os.listdir(fileDir)
 #  for image_name in image_set_old:
 #       old_name=os.path.join(fileDir,image_name)
 #       new_name=old_name[:-4]+'.png'
 #       os.rename(old_name,new_name)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(test_xml_dir):
        os.makedirs(test_xml_dir)
    if not os.path.exists(train_xml_dir):
        os.makedirs(train_xml_dir)
    if not os.path.exists(train_nooa_dir):
        os.makedirs(train_nooa_dir)
    if not os.path.exists(test_nooa_dir):
        os.makedirs(test_nooa_dir)
        
        
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.1
    number = int(filenumber * rate)           # 按照rate比例从文件夹中取数据
    sample = random.sample(pathDir, number)  # 随机选取picknumber数量的数据
    #print (sample)
    for name in sample:#测试集
        shutil.copy(fileDir + name, test_dir + name)
        #shutil.copy(fileNooaDir + name.split('.')[0]+'.png', test_nooa_dir + name.split('.')[0]+'.png')
        shutil.copy(fileNooaDir + name[:-4]+".png", test_nooa_dir + name[:-4]+".png")
        item = name[:-4]
        shutil.copy(xml_dir+item+'.xml',test_xml_dir+item+'.xml')

    for filename in os.listdir(fileDir):#训练集
        if filename not in os.listdir(test_dir):
            shutil.copy(os.path.join(fileDir,filename),os.path.join(train_dir,filename))
            #shutil.copy(os.path.join(fileNooaDir,filename[:-4]+'.png'),os.path.join(train_nooa_dir,filename[:-4]+'.png'))
            shutil.copy(os.path.join(fileNooaDir,filename[:-4]+".png"),os.path.join(train_nooa_dir,filename[:-4]+".png"))
            shutil.copy(xml_dir + filename[:-4]+ '.xml',train_xml_dir + filename[:-4] + '.xml')
    return
if __name__ == '__main__':
    data_base = './dataset/'
    moveFile(data_base)
