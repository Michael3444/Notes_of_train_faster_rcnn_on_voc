# 训练faster rcnn的一些笔记
[Faster RCNN](https://github.com/DetectionTeamUCAS/Faster-RCNN_Tensorflow)的训练过程，结合自己对代码的理解，在此记录下来。此笔记并不会讨论具体的代码细节。 

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


## 2.generate anchors
网络结构的backbone使用的是resnet101，rpn网络的输入是resnet101第3个stage的输出，rpn先经过一个3x3的卷基层，该层不会改变feature map的大小，不过会影响该层感受野的大小。我计算的rpn经过一个3x3的卷基层后，感受野大小为811(这个值是理论感受野的大小，有效感受野的大小会比该值小很多)，不过感受野中心的位置是和图片的尺寸有关的，代码中的anchors的中心是选取的（0，0）（0，16）等一系列固定值而不是感受野中心，我猜这么处理是为了计算方便，不用每一个batch都计算一次中心位置，而且本身所有的anchors也是人为选择的，中心位置带来的误差并不会对最终的结果带来很坏的影响。

## 3.Label the anchors 
这个代码的训练是一个端到端的过程，并不是rpn和fast rcnn交替训练，所以每一个迭代过程，都会对生成的anchors进行标注，规则如下：
* anchors被分为三类：1代表前景，0代表背景，-1代表忽略此框。
* 被标注的anchors必须位于图像内部，超出图像边界的anchors被排除在外。
* 与所有ground truth的iou最大的anchor，标记为正样本，label=1。
* anchor与某一个ground truth的iou大于0.7，标记为正样本，label=1。
* anchor与所有的ground truth的iou都小于0.3，标记为负样本，label=0。
* 剩余样本既不是正样本，也不是负样本，不参与训练。
* 确保前景anchors的数量fg_nums不超过128个，超出的的全标注为-1。
* 确保背景anchors的数量不超过256-fg_nums，将超出的全标记为-1。
* 所有的在图像内部的anchors与和其iou最大的ground truth进行编码，超出图像范围的anchors设为0。
## 4.
****

****
 
