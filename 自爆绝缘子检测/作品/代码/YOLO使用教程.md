### 教程

1. [下载权重文件yolov3.weights]('https://pjreddie.com/media/files/yolov3.weights')

 2.  将yolov3.weights放到根目录中的temp文件夹下，执行命令进行转换：python convert.py yolov3.cfg yolov3.weights ../model_data/yolo_weight.h5

     ![image-20200507193236895](img\image-20200507193236895.png)

3. 先运行kmean2生成锚框信息写入model_data/yolo_anchors.txt中

4. 然后将train.txt中的信息填好

5. train.py中的input_shape = (128,2048)信息更换好，这里值训练集图片的分辨率，需要是32的倍数，运行train.py训练好模型，logs/000中得到的最好模型路径替换到yolo_defect的"model_path"中

   ![捕获](img\捕获.PNG)

6. 最后可以开始测试，运行test

   - 将测试集图片放置在keras-yolo3-insulator\dataset\test_insulator
   - 运行过程如下图

   > ![捕3](img\捕3.PNG)

