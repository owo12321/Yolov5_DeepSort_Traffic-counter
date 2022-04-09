# Yolov5_DeepSort_Traffic-counter
基于Yolov5_DeepSort的移动物体计数器，可以统计车流或人流量等  
本作品基于此项目实现：https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch 在它的基础上做了一些改动，实现了统计画面中经过的物体数量的功能，包括车流、人群等。
![image](https://github.com/owo12321/Yolov5_DeepSort_Traffic-counter/blob/main/test.gif)

## 1、环境配置
下载项目文件夹后，在命令行中进入项目文件夹，执行以下代码配置环境：
```
pip install -r requirements.txt
```

## 2、检测原理
Yolov5_DeepSort会跟踪画面上检测出来的物体，并给每个框标上了序号，当有一个方框跨过检测线时，计数器就会+1  
用户可以指定检测线的画法，可以指定框沿哪个方向跨过检测线时计数器+1，也可以指定框的四个顶点中的哪一个顶点跨过线时计数器+1  
具体的参数设定见第3点


## 3、参数设置
在count.py中，设置以下参数
```
source_dir : 要打开的视频文件。若要调用摄像头，需要设置为字符串'0'，而不是数字0，按q退出播放
output_dir : 要保存到的文件夹
show_video : 运行时是否显示
save_video : 是否保存运行结果视频
save_text :  是否保存结果数据到txt文件中，将会保存两个文本文件：result.txt和number.txt。result.txt的格式是(帧序号,框序号,框到左边距离,框到顶上距离,框横长,框竖高,-1,-1,-1,-1)，number.txt的格式是(帧序号，直至当前帧跨过线的框数)

class_list : 要检测的类别序号，在coco_classes.txt中查看（注意是序号不是行号），可以有一个或多个类别

line : 检测线的两个端点的xy坐标，总共4个数
big_to_small : 0表示从比线小的一侧往大的一侧，1反之。（靠近坐原点或坐标轴的一侧是小的）
point_idx : 要检测的方框顶点号(0, 1, 2, 3)，看下边的图，当方框的顶点顺着big_to_small指定的方向跨过检测线时，计数器会+1
```

检测线的画法：给出两个端点的坐标，确定一条检测线，画布的坐标方向如下
```
   |-------> x轴
   |
   |
   V
   y轴
```

方框的顶点编号：当方框的指定顶点跨过检测线时，计数器会+1，顶点的编号如下
```
   0              1
   |--------------|
   |              |
   |              |
   |--------------|
   3              2
```

## 4、运行
设置好参数后，python运行count.py文件即可
```
python count.py
```
