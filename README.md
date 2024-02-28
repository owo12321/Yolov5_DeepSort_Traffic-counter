# Yolov5_DeepSort_Traffic-counter
基于Yolov5_DeepSort的移动物体计数器，可以统计车流或人流量等  
本作品基于此项目实现：https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch  
实现了统计画面中通过检测线的物体数量的功能，包括车流、人群等。

![image](https://github.com/owo12321/Yolov5_DeepSort_Traffic-counter/blob/main/test3.gif)

## 0、更新
1.可以绘制多条检测线  
2.每条检测线可以同时统计两个跨线方向的流量  

## 1、环境配置
下载项目文件夹后，在命令行中进入项目文件夹，执行以下代码配置环境：
```
pip install -r requirements.txt
```
在Yolov5_DeepSort_Traffic-counter/deep_sort_pytorch/deep_sort/deep/checkpoint路径下需要下载一个文件
```
链接：https://pan.baidu.com/s/1BwMUM9JGRhMQgmjTu_HXcw?pwd=bwux 
提取码：bwux 
```
默认使用Yolov5的5.0版本的yolov5s.pt模型文件，建议训练自己的数据集，参考  
https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data


## 2、检测原理
Yolov5_DeepSort会跟踪画面上检测出来的物体，并给每个框标上了序号，当有一个方框跨过检测线时，计数器就会+1  
用户可以指定检测线的起点终点坐标，也可以指定框的四个顶点或中心点哪一个作为检测点  
具体的参数设定见第3点


## 3、参数设置
在count.py中，设置以下参数
```
source_dir : 要打开的视频文件。若要调用摄像头，需要设置为字符串'0'，而不是数字0，按q退出播放
output_dir : 要保存到的文件夹
show_video : 运行时是否显示
save_video : 是否保存运行结果视频
save_text :  是否保存结果数据到txt文件中，将会保存两个文本文件：result.txt和number.txt。result.txt的格式是(帧序号,框序号,框到左边距离,框到顶上距离,框横长,框竖高,-1,-1,-1,-1)，number.txt的内容是统计到第几帧时每条线沿两个方向的跨线物体数

class_list : 要检测的类别序号，在coco_classes.txt中查看（注意是序号不是行号），可以有一个或多个类别

lines : 定义检测线的两个端点的xy坐标、颜色、粗细，可以定义多条检测线
point_idx : 方框的检测点位置(0, 1, 2, 3, 4)，看下边的图，当一个方框的检测点跨过检测线时，统计数会+1
```

检测线的画法：给出两个端点的坐标，确定一条检测线，画布的坐标方向如下
```
   |-------> x轴
   |
   |
   V
   y轴
```

方框的检测编号：当一个框的检测点跨过检测线时，计数器会+1，检测点的编号如下
```
   1__________________2
   |                  |
   |                  |
   |      0(中心点)   |
   |                  |
   |__________________|
   4                  3
```

## 4、运行
设置好参数后，python运行count.py文件即可
```
python count.py
```
