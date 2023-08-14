# fallen-person-recognize
## 新内容

2023.8 —— 支持实时多人姿态估计，**33FPS** on Apple M1Pro。

 2023.7 —— 添加**关键点行为识别数据集**，共110000+带有标注的关键点数据。[Google Drive](https://drive.google.com/drive/folders/1-n0jYog_vLufOdzq5lYgvuI1q_ulrpD8?usp=drive_link)

 2023.7 —— 支持**多人姿态估计**

 2023.2 —— 支持**实时单人姿态估计**

## TODO
- [x] 基于COCO数据集重新训练姿态估计模型并调参（**47FPS**）。
- [x] 基于PicoDet的实时人体检测器（**73FPS**）。
- [ ] 基于ByteTrack的姿态跟踪。
  - [x] 检测框跟踪。
  - [ ] 骨架跟踪。
- [ ] 基于深度学习的行为识别模型。
  - [x] 数据集（NTU-120）。
  - [x] 主流模型复现。
  - [ ] 轻量级模型搭建。
  - [ ] 模型改进。
- [ ] 优化模型性能。
  - [ ] 骨骼点抖动处理。
  - [ ] PicoDet优化。
  - [ ] Pipeline优化。
  - [ ] Tracker优化。
- [ ] 发布Python的部署版本。
- [ ] 发布C++的部署版本。
- [ ] 计算并行化。
## 说明
Config→[Config](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/docs/config.md)


分支→[Branch](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/docs/branch.md)


项目结构→[Files](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/docs/structure.md)

## 成果

### 基于骨骼的行为识别模型对比

| 模型                                             | 复现准确率(%) | 论文准确率(%)                                     | 推理速度(ms) | 参数量(M) |
| ------------------------------------------------ | ------------- | ------------------------------------------------- | ------------ | --------- |
| [ST-GCN](https://arxiv.org/pdf/1801.07455v2.pdf) | 85.8          | [88.8](https://arxiv.org/pdf/1801.07455v2.pdf) | 79.2  ± 4.4  | 3.095     |
|**TODO** [SGN](https://arxiv.org/pdf/1904.01189.pdf)|-|-|-|-|
|**TODO** [STID](https://arxiv.org/pdf/2208.05233.pdf)|-|-|-|-|
### 模型性能对比

| 模型                                              | FLOPS(G) | 参数量(M) | 平均推理耗时(ms/frame)`*` | 说明                                                                                           |
| ------------------------------------------------- | -------- | --------- | ------------------------- | ---------------------------------------------------------------------------------------------- |
| MFNet                                             | 0.67     | 4.10      | 21/人                     | 姿态估计模型                                                                                   |
| [PicoDet](https://arxiv.org/pdf/2111.00902.pdf)   | 1.18     | 0.97      | 13.7                      | 人体检测模型                                                                                   |  |
| [ByteTrack](https://arxiv.org/pdf/2110.06864.pdf) | -        | -         | 7.3                       | 目标跟踪模型,**TODO**                                                                          |
| 共计                                              | 1.85     | 5.07      | 42                        | [推理视频](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/video.mov) |

`*`说明: 基于M1Pro(8+2)平台推理。

### 行为识别数据集
下载:[Google Drive](https://drive.google.com/drive/folders/1-n0jYog_vLufOdzq5lYgvuI1q_ulrpD8?usp=drive_link)(主要是不限速)  [百度网盘](https://pan.baidu.com/s/1Mw040S7RUPSiRFxxCGgxZA?pwd=p7sc)

![结果](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/ST-GCN_Skeleton.jpg)

### 多人-遮挡 
![足球](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/multi-pose-estimation.png)

 检测器：[PicoDet](https://arxiv.org/pdf/2111.00902.pdf)
 姿态估计模型： **MFNet** 

### 坐卧

![坐姿](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/sit-pose-estimation.png)
 检测器：[PicoDet](https://arxiv.org/pdf/2111.00902.pdf)
 姿态估计模型： **MFNet** 

### 站立

![站姿](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/stand-pose-estimation.png)
 检测器：[PicoDet](https://arxiv.org/pdf/2111.00902.pdf)
 姿态估计模型： **MFNet** 

### 广角
![广角](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/wide_angle1.jpg)
 检测器：[PicoDet](https://arxiv.org/pdf/2111.00902.pdf)
 姿态估计模型： **MFNet** 

## 规划
### 阶段一
完成对姿态估计模型的建立、训练，确定最终落地使用的姿态估计方案。
  
要求：**AP: 0.80+**,**FPS: 18+** 
### 阶段二
基于阶段一的姿态估计方案，使用姿态估计数据，建立数学模型或轻量级神经网络模型，实现跌倒识别。
  
要求：**Accuracy: 0.95+**, **CPU-REALTIME**

### 阶段三
基于`micro-python`,`Django`,`Flutter`实现模型落地部署。其中，`micro-python`用于在边缘设备记录图像信息，并上传至服务器;`Django`用于后端搭建，基于`pytorch`实现图像推理，并得到识别结果;`Flutter`用于搭建跨平台的App,用于向用户展示识别结果信息（如是否跌倒、姿态信息等）。
## Reference
+ [TCFormer](https://arxiv.org/pdf/2204.08680.pdf)
+ [Simple BaseLine](https://arxiv.org/pdf/1804.06208.pdf)
+ [HRNet](https://arxiv.org/pdf/1902.09212.pdf)
+ [MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)
+ [YOLOV7](https://arxiv.org/pdf/2207.02696.pdf)
+ [YOLOV5](https://github.com/ultralytics/yolov5)
+ [PicoDet](https://arxiv.org/pdf/2111.00902.pdf)
+ [ST-GCN](https://arxiv.org/pdf/1801.07455v2.pdf) 
+ [ByteTrack](https://arxiv.org/pdf/2110.06864.pdf)