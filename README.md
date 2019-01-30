# 训练faster rcnn的一些笔记
[Faster RCNN](https://github.com/DetectionTeamUCAS/Faster-RCNN_Tensorflow)的训练过程，结合自己对代码的理解，在此记录下来。 

****
|Author|Michael|
|---|---|
|E-mail|1458319689@qq.com|
****

## 1.data preparation
transfer raw image files into tfrecord format   
```
{
  'img_name': '000059.jpg',
  'img_height': 500,
  'img_width': 375,
  'img': a numpy array,
  'gtboxes_and_label': [[x1,y1,x2,y2,label]...],
  'num_objects': 6
}
```
训练的batchsize设为1，图片的短边设为固定值600， 长边按比例resize， 注意最大边不超过1200，图像随机翻转。  


## 2. generate anchors
网络结构的backbone使用的是resnet101，rpn网络的输入是resnet第3个stage的输出，rpn先经过一个3x3的卷基层，该层不会改变feature map的大小，不过会影响该层感受野的大小，我计算经过改3x3的卷基层后，感受野大小为811，不过感受野中心的位置是和图片的尺寸有关的，代码中的anchors的中心是选取的（0，0）（0，16）等一系列固定值而不是感受野中心，我猜这么处理是为了计算方便，不用每一个batch都计算一次中心位置，而且本身所有的anchors也是人为选择的，中心位置带来的误差并不会对最终的结果带来很坏的影响。

## 3.
## 4.
****

****
 
