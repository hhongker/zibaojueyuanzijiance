## 教程

1. - 将原图放置dataset/init/val/

   - 将掩模图放置dataset/init/val_anno/
   - 将xml放置dataset/init/xml

2. 运行顺序：Split > main_imgaug > main_block > main_segment > main_predict > main_predict_noise > main_predict_pca > main_evalate_dice

3. 每个代码的详解

   + split：划分训练和验证样本，并保存到不同目录。

   + main_imgaug：对训练样本增强，并保存到不同目录。

   + main_block：对增强后的训练样本划分小块，713*713像素为一块，不足的补0。掩模对应切割。对验证样本也切块，保存到不同目录。

     > **注意**：这里main_block先对训练集进行分块，然后对测试集进行分块。对训练集分块时，将测试集合的路径注释掉。对测试集分块时同理，也就是说main_block需要运行两次
     >
     > ![捕获2](img\捕获2.PNG)
   
   - main_segment：基于PSPNet，用训练样本的子块训练，用验证样本的子块验证。得到训练好的语义分割模型。
   - main_predict：对某个测试样本的原图，切分子块，输入到PSPNet中，对每个子块分别得到掩模，然后并组装成一个测试样本完整的掩模。
   - main_predict_noise：降噪，求连通区域的坐标的均值点，和坐标点集合的PCA的两个方向，求坐标集合在分别在两个方向上的投影。得到投影的最小值和最大值的差，即可得到连通区域的长度和宽度（若长度和宽度之比小于2，则认为是噪音，忽略）。在PCA1长度方向上每隔2倍的宽度作为边长进行均匀采样图像的正方形子块，步长为1倍宽度，即重叠一半。相应的自爆点方框坐标也进行相应的平移，以获得正方形子块里自爆点的坐标。
   - main_predict_pca：基于主成分的旋转和裁剪的规范化绝缘子区域算法。将每张图片的每串绝缘子获取出来，独立成像。规范化的操作。为后面yolo模型做准备。
   - main_evalate_dice：对测试得到的数据进行dice系数测评