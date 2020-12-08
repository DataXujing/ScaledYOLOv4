'''

inference single image

xujing

reference detect.py

'''


import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from tqdm import tqdm

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import shutil

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

# 超参数
conf_thres = 0.4       # NMS中的概率阈值
iou_thres = 0.5          # NMS的IoU阈值
merge = True             # NMS中是否boxes merged using weighted mean
# prob_thres = 0.2        # 最后展示框的概率，在NMS后, 0.3, 0.5, 0.6, 0.7,0.75, 0.8, 0.85, 0.9
weights = "./runs/exp0_yolov4-p7/weights/best.pt"   # 模型权重路径
input_size = 640

names = ["Liomyoma", "Lipoma", "Pancreatic Rest", "GIST", "Cyst",  "NET", "Cancer"]


test_dir = "./eus/images/val"

device = select_device("0", batch_size=1)

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(input_size, s=model.stride.max())  # check img_size

# config
model.eval()
# with open("./data/bing.yaml") as f:
#     data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
# nc = 1 if single_cls else int(data['nc'])  # number of classes

# 加载图像
files = [file for file in os.listdir(test_dir) if ".xml" not in file]

filebar = tqdm(files)

for file in filebar:
    if ".xml" in file:
        continue

    img_path = os.path.join(test_dir,file)
    # 对图像，变成tensor
    img0 = cv2.imread(img_path)
    # Padded resize
    img = letterbox(img0, new_shape=input_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)

    # torch tensor的操作
    img = torch.from_numpy(img).to(device)
    img = img.float()
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # nb, _, height, width = img.shape  # batch size, channels, height, width
    # whwh = torch.Tensor([width, height, width, height]).to(device)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=True)[0]   # 使用TTA

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
    t2 = time_synchronized()


    # Process detections

    for i, det in enumerate(pred):  # detections per image
        # if webcam:  # batch_size >= 1
        #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
        # else:
        #     p, s, im0 = path, '', im0s

        # save_path = str(Path(out) / Path(p).name)
        # txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        # s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            # for c in det[:, -1].unique():
            #     n = (det[:, -1] == c).sum()  # detections per class
            #     s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            # det_count = 0
            for *xyxy, conf, cls_ in det:   # x1,y1,x2,y2
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     with open(txt_path + '.txt', 'a') as f:
                #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                # if save_img or view_img:  # Add bbox to image
                    # label = '%s' % (names[int(cls)])
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)




                label = '%s' % (names[int(cls_)])
                label_text = label
                # print(conf.cpu().detach().numpy())
                prob = round(conf.cpu().detach().numpy().item(),2)


                # tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

                color = (255, 255, 0)
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                cv2.rectangle(img0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label_text+":"+str(prob), 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(img0, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img0, label_text+":"+str(prob), (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    if not os.path.exists("./detect_res"):
        os.makedirs("./detect_res")
    cv2.imwrite("./detect_res/"+file,img0)

               


    filebar.set_description("[INFO] 正在处理:{},FPS: {}".format(file,1/(t2-t1)))







