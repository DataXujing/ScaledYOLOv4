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
# files = [file for file in os.listdir(test_dir) if ".xml" not in file]

# filebar = tqdm(files)

video_names = os.listdir("./test_vid")

for video_name_ in video_names:

    video_name = "./test_vid/" + video_name_

    # 识别图片的保存结果
    save_path = video_name_.split(".")[0]
    if not os.path.exists("./video/"+save_path):
        os.makedirs("./video/"+save_path)


    cap = cv2.VideoCapture(video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_video = cv2.VideoWriter(os.path.join("./video/",video_name_),fourcc,fps,size)



    i_frame = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        i_frame += 1

        if ret == True:

            # 对图像，变成tensor
            # img0 = cv2.imread(img_path)
            img0 = frame.copy()
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

            det_count = 0

            for i, det in enumerate(pred):  # detections per image
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    for *xyxy, conf, cls_ in det:   # x1,y1,x2,y2
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        # if save_img or view_img:  # Add bbox to image
                            # label = '%s' % (names[int(cls)])
                            # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)


                        label = '%s' % (names[int(cls_)])

                        # if not os.path.exists("./metric/detections"):
                        #     os.makedirs("./metric/detections")

                        if label in ["A","B","C","D","E","N1","N2","N3","N4","N5","N6","N7","N8","N9","N10"]:
                            continue
                        if conf <= prob_thres:
                            continue

                        det_count += 1

                        label_text = names2label[label]
                        # print(conf.cpu().detach().numpy())
                        prob = round(conf.cpu().detach().numpy().item(),2)


                        # tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

                        color = (255, 255, 0)
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                        cv2.rectangle(img0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                        tf = max(tl - 1, 1)  # font thickness
                        t_size = cv2.getTextSize(label_text+":"+str(prob), 0, fontScale=tl / 1.5, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.rectangle(img0, c1, c2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(img0, label_text+":"+str(prob), (c1[0], c1[1] - 2), 0, tl / 1.5, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

            out_video.write(img0)

            if det_count >= 1:
                cv2.imwrite( "./video/{}/{}.jpg".format(save_path,str(i_frame)), img0 )

            # cv2.imshow("yolov4-p7",img0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


    cap.release()
    out_video.release()
    cv2.destroyAllWindows()







