## [Scaled YOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4) 训练自己的数据集

> 感谢 <https://github.com/WongKinYiu/ScaledYOLOv4> 大佬的开源！！！

**DataXujing**

我们以训练YOLOv4-P7为例，介绍如何基于Scaled YOLOv4训练自己的数据集

### 0.环境配置

```
python3.7 cuda 10.2
pytorch==1.6.0
torchvision==0.7.0

# mish-cuda
# 使用预训练的模型
git clone https://github.com/thomasbrandon/mish-cuda mc
cd mc

# change all of name which is mish_cuda to mish_mish and build.
# 1. mc/src/mish_cuda -> mc/src/mish_mish
# 2. mc/csrc/mish_cuda.cpp -> mc/csrc/mish_mish.cpp
# 3. in mc/setup.py
#   3.1 line 5 -> 'csrc/mish_mish.cpp'
#   3.2 line 11 -> name='mish_mish'
#   3.3 line 20 -> 'mish_mish._C'

python setup.py build
# rename mc/build/lib.xxx folder to mc/build/lib

# modify import in models/common.py
# line 7 -> from mc.build.lib.mish_mish import MishCuda as Mish
# 不使用预训练的模型，可以
git clone https://github.com/thomasbrandon/mish-cuda
cd mish-cuda
python setup.py build install

```

遗憾的是我再CUDA9.0下成功的安装了mish-cuda,但是在CUDA9.2,CUDA10.0和CUDA10.2下均未成功安装mish-cuda,最后我在common.py中基于pytorch实现了Mish,实现方式如下：

```python
 no pretrain
# from mish_cuda import MishCuda as Mish
# pretrain
#from mc.build.lib.mish_mish import MishCuda as Mish

#------------------by xujing------------
# 实在装不上mish-cuda,用pytorch自己实现
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        # print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
# ---------------------------------------------


```

### 1.数据集准备

数据集的准备请参考：<https://github.com/DataXujing/YOLO-v5>,其结构如下图所示：

```
eus         # 数据集名称
├─images
│  ├─train  # 训练图片存放地址
│  └─val    # 验证图片存放地址
└─labels
    ├─train # train txt标注文件存放地址
    └─val   # val txt标注文件存放地址

```

### 2.模型修改

+ 修改增加`./data/eus.yaml`

```
# train and val datasets (image directory or *.txt file with image paths)
train: ./eus/images/train  # <-----------训练集存放地址
val: ./eus/images/val  # <----------开发集存放地址
test: ./eus/images/val  # <--------测试集存放地址

# number of classes
nc: 7     # <----------类别数量

# class names
names: ["Liomyoma", "Lipoma", "Pancreatic Rest", "GIST", "Cyst",  "NET", "Cancer"]   # <----列别列表


```

+ 修改增加`./models/eus/yolov4-p7.yaml`

```
# parameters
nc: 7  # <----------------------------------number of classes
depth_multiple: 1.0  # expand model depth
width_multiple: 1.25  # expand layer channels

# anchors
anchors:
  - [13,17,  22,25,  27,66,  55,41]  # P3/8
  - [57,88,  112,69,  69,177,  136,138]  # P4/16
  - [136,138,  287,114,  134,275,  268,248]  # P5/32
  - [268,248,  232,504,  445,416,  640,640]  # P6/64
  - [812,393,  477,808,  1070,908,  1408,1408]  # P7/128

# csp-p7 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [32, 3, 1]],  # 0
   [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
   [-1, 1, BottleneckCSP, [64]],
   [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4
   [-1, 3, BottleneckCSP, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 5-P3/8
   [-1, 15, BottleneckCSP, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 7-P4/16
   [-1, 15, BottleneckCSP, [512]],
   [-1, 1, Conv, [1024, 3, 2]], # 9-P5/32
   [-1, 7, BottleneckCSP, [1024]],
   [-1, 1, Conv, [1024, 3, 2]], # 11-P6/64
   [-1, 7, BottleneckCSP, [1024]],
   [-1, 1, Conv, [1024, 3, 2]], # 13-P7/128
   [-1, 7, BottleneckCSP, [1024]],  # 14
  ]

# yolov4-p7 head
# na = len(anchors[0])
head:
  [[-1, 1, SPPCSP, [512]], # 15
   [-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [-6, 1, Conv, [512, 1, 1]], # route backbone P6
   [[-1, -2], 1, Concat, [1]],
   [-1, 3, BottleneckCSP2, [512]], # 20 
   [-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [-13, 1, Conv, [512, 1, 1]], # route backbone P5
   [[-1, -2], 1, Concat, [1]],
   [-1, 3, BottleneckCSP2, [512]], # 25
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [-20, 1, Conv, [256, 1, 1]], # route backbone P4
   [[-1, -2], 1, Concat, [1]],
   [-1, 3, BottleneckCSP2, [256]], # 30
   [-1, 1, Conv, [128, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [-27, 1, Conv, [128, 1, 1]], # route backbone P3
   [[-1, -2], 1, Concat, [1]],
   [-1, 3, BottleneckCSP2, [128]], # 35
   [-1, 1, Conv, [256, 3, 1]],
   [-2, 1, Conv, [256, 3, 2]],
   [[-1, 30], 1, Concat, [1]],  # cat
   [-1, 3, BottleneckCSP2, [256]], # 39
   [-1, 1, Conv, [512, 3, 1]],
   [-2, 1, Conv, [512, 3, 2]],
   [[-1, 25], 1, Concat, [1]],  # cat
   [-1, 3, BottleneckCSP2, [512]], # 43
   [-1, 1, Conv, [1024, 3, 1]],
   [-2, 1, Conv, [512, 3, 2]],
   [[-1, 20], 1, Concat, [1]],  # cat
   [-1, 3, BottleneckCSP2, [512]], # 47
   [-1, 1, Conv, [1024, 3, 1]],
   [-2, 1, Conv, [512, 3, 2]],
   [[-1, 15], 1, Concat, [1]],  # cat
   [-1, 3, BottleneckCSP2, [512]], # 51
   [-1, 1, Conv, [1024, 3, 1]],

   [[36,40,44,48,52], 1, Detect, [nc, anchors]],   # Detect(P3, P4, P5, P6, P7)
  ]

```



### 3.模型训练

```
# {YOLOv4-P5, YOLOv4-P6, YOLOv4-P7} use input resolution {896, 1280, 1536} for training respectively.
# 我在V100上训练，但是对于P7而言，在896的分辨率下，batch size=4才可以正常训练

python37 -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 896 896 --data eus.yaml --cfg models/eus/yolov4-p7.yaml --weights 'pretrain/yolov4-p7.pt' --sync-bn --device 0 --name yolov4-p7

python37 -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 64 --img 896 896 --data eus.yaml --cfg models/eus/yolov4-p7.yaml --sync-bn --device 0 --name yolov4-p7


python train.py --batch-size 64 --img 896 896 --data eus.yaml --cfg models/eus/yolov4-p7.yaml --weights 'pretrain/yolov4-p7.pt' --sync-bn --device 0 --name yolov4-p7

python --nproc_per_node 4 train.py --batch-size 64 --img 896 896 --data eus.yaml --cfg models/eus/yolov4-p7.yaml --weights 'pretrain/yolov4-p7.pt' --sync-bn --device 0 --name yolov4-p7
python train.py --batch-size 64 --img 896 896 --data eus.yaml --cfg models/eus/yolov4-p7.yaml --weights '' --sync-bn --device 0 --name yolov4-p7

# V100下的训练
python train.py --batch-size 4 --img 896 896 --data eus.yaml --cfg models/eus/yolov4-p7.yaml --weights '' --sync-bn --device 0 --name yolov4-p7

python train.py --batch-size 8 --img 640 640 --data eus.yaml --cfg models/eus/yolov4-p7.yaml --weights '' --sync-bn --device 0 --name yolov4-p7

```


### 4.模型推断

+ 模型测试

```
# download {yolov4-p5.pt, yolov4-p6.pt, yolov4-p7.pt} and put them in /yolo/weights/ folder.
python test.py --img 896 --conf 0.001 --batch 8 --device 0 --data eus.yaml --weights weights/yolov4-p5.pt
python test.py --img 1280 --conf 0.001 --batch 8 --device 0 --data eus.yaml --weights weights/yolov4-p6.pt
python test.py --img 1536 --conf 0.001 --batch 8 --device 0 --data coco.yaml --weights weights/yolov4-p7.pt

```

+ 模型推断

**TODO**


### 5.DEMO

**TODO**

### 6.TensorRT加速

**TODO**


