# yolov5 目标检测+计数
---

* 实现目标检测中的 图片/视频 计数
* 显示检测的类别
* to do

###效果展示

![Alt text](https://cdn.nlark.com/yuque/0/2023/jpeg/35529404/1680766431805-e45162a1-f84e-46a4-9c9b-5758aa79894d.jpeg?x-oss-process%3Dimage%2Fresize%2Cw_937%2Climit_0%2Finterlace%2C1)

### 一、运行环境 
* ubuntu 18.04
* python 3.8
* pytorch 1.11

### 二、环境安装
2.1、下载代码 进入相应目录
```
git clone https://github.com/koihoo/yolov5_counting_pytorch.git
cd yolov5_counting_pytorch
```
2.2、conda创建虚拟环境 并激活环境
```
conda create -n yolov5 python=3.8
conda activate yolov5
```
2.3、安装pytorch
> 根据操作系统、安装工具以及CUDA版本，在 https://pytorch.org 找到对应的安装命令。我的环境是 ubuntu 18.04.5、pip、CUDA 11.3
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
2.4、安装必要的软件包
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### 三、数据准备——训练自己的数据集
3.1 原始数据格式(插入yolov5/data文件夹中)
```
dataset #(数据集名字：例如goose) 
├── images      
       ├── train          
              ├── xx.jpg     
       ├── val         
              ├── xx.jpg 
├── labels      
       ├── train          
              ├── xx.txt     
       ├── val         
              ├── xx.txt 
```

在yolov5/data文件夹下新建goose.yaml

```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]


path: /home/koihoo/yolov5/data/goose-2 # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: ['goose']  # class names
```

>path：数据集的根目录
train：训练集与path的相对路径
val：验证集与path的相对路径
nc：类别数量，因为这个数据集只有一个类别（fire），nc即为1。
names：类别名字

3.2下载预训练模型	
根据需求选择https://github.com/ultralytics/yolov5/releases
模型下载完成后，将xx.pt复制在yolov5文件夹下。

### 四、训练
上面的数据和预训练模型都准备好之后，我们就可以开始训练啦
```
python train.py --weights yolov5s.pt --data data/goose.yaml --workers 1 --batch-size 8
```
训练最终结果如图所示

![output](https://cdn.nlark.com/yuque/0/2023/jpeg/35529404/1680766052667-3ea7e7b9-f712-4bca-97a7-98b52944449f.jpeg)

> 训练窗口中添加需要打印的指标（如F1-score）
 
在yolov5/val.py中的第192行开始
```python
# L 192 ,注意参数量，从7变成8
s = ('%22s' + '%11s' * 7) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95', 'F1-score')

# L 285
mp, mr, map50, map, f1 = p.mean(), r.mean(), ap50.mean(), ap.mean(), f1.mean()

# L 292
pf = '%22s' + '%11i' * 2 + '%11.3g' * 5  # print format
LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map, f1))
```

### 五、测试
模型训练完成之后会保存在runs/train，使用在自己数据集上训练好的模型可以进行预测
```
python detect.py --weights runs/train/exp/weights/best.pt --source xxx.mp4 --view-img --hide-conf 
```

> 在yolov5/detect.py中加入counting的功能代码

```python\
# L 207
####### COUNTING ########
cv2.putText(im0,f"{names[int(c)]}{'s' * (n > 1)}: {n}", (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
```

---
# 使用框架
* https://github.com/ultralytics/yolov5/ 