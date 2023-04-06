import cv2
import torch
from numpy import random

# YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()  

# 定义类别列表
classes = ['goose']

# 定义计数器
counter = [0] * len(classes)

# 视频文件路径
video_path = 'test.mp4'

# 视频帧处理函数
def process_frame(frame):
    # 运行YOLOv5检测
    results = model(frame)

    # 统计每个类别的数量
    for res in results.pred:
        for i, c in enumerate(res[:, -1].unique()):
            n = (res[:, -1] == c).sum()  # 计算每个类别的数量
            counter[int(c)] += n  # 增加计数器

    # 在视频帧上添加文本框和文本
    for i, c in enumerate(classes):
        cv2.putText(frame, f"{c}: {counter[i]}", (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

# 打开视频文件
cap = cv2.VideoCapture(video_path)

while True:
    # 读取视频帧
    ret, frame = cap.read()

    if not ret:
        break

    # 处理视频帧
    frame = process_frame(frame)

    # 显示视频帧
    cv2.imshow('frame', frame)

    # 按下q键退出
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
