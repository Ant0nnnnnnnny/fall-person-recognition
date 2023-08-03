# fallen-person-recognize
## 新内容

 2023.7 —— 添加**关键点行为识别数据集**，共110000+带有标注的关键点数据。[Google Drive](https://drive.google.com/drive/folders/1-n0jYog_vLufOdzq5lYgvuI1q_ulrpD8?usp=drive_link)

 2023.7 —— 支持**多人姿态估计**

 2023.2 —— 支持**实时单人姿态估计**
## 说明
Config→[Config](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/docs/config.md)


分支→[Branch](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/docs/branch.md)


项目结构→[Files](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/docs/structure.md)

## 成果

### 模型性能对比

| 模型                                                   | Macs(M) | 参数量(M) | 平均推理耗时(ms/frame) | 说明                                                                                           |
| ------------------------------------------------------ | ------- | --------- | ---------------------- | ---------------------------------------------------------------------------------------------- |
| MFNet                                                  | 1035.96 | 4.10      | 21/人                  | 姿态估计模型                                                                                   |
| [FastestDet](https://github.com/dog-qiuqiu/FastestDet) | 779.38  | 0.24      | 61                     | 人体检测模型                                                                                   |
| [ST-GCN](https://github.com/hazdzz/STGCN)              | 55.86   | 2.62      | 21                     | 动作分类模型**TODO**                                                                           |
| 共计                                                   | 1871.2  | 6.96      | 102                    | [推理视频](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/video.mov) |

### 行为识别数据集
下载:[Google Drive](https://drive.google.com/drive/folders/1-n0jYog_vLufOdzq5lYgvuI1q_ulrpD8?usp=drive_link)(主要是不限速)  [百度网盘](https://pan.baidu.com/s/1Mw040S7RUPSiRFxxCGgxZA?pwd=p7sc)

![结果](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/skeleton-dataset.jpg)

### 多人-遮挡 
![足球](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/multi-pose-estimation.png)

 检测器：[YOLOS-base](https://huggingface.co/hustvl/yolos-base)
 姿态估计模型： **MFNet** 

### 坐卧

![坐姿](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/sit-pose-estimation.png)
 检测器：[YOLOS-tiny](https://huggingface.co/hustvl/yolos-tiny) 
 姿态估计模型： **MFNet** 

### 站立

![站姿](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/stand-pose-estimation.png)
 检测器：[YOLOS-tiny](https://huggingface.co/hustvl/yolos-tiny) 
 姿态估计模型： **MFNet** 

### 跌倒
![跌倒](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/fallen-pose-estimation.png)
 检测器：[YOLOS-tiny](https://huggingface.co/hustvl/yolos-tiny) 
 姿态估计模型： **MFNet** 


<!-- # 当前模型对比
| 模型名称           | Params     | MACs       | AP       | FPS      |
| ------------------ | ---------- | ---------- | -------- | -------- |
| MobileNetV3-normal | **5.285M** | **3.814G** | 0.65     | **28.5** |
| MobileNetV3-large  | 10.871M    | 4.053G     | 0.76     | 22.7     |
| TCFormer           | 25.624M    | 6.535G     | **0.82** | 9.4      |
| **MFNet**-normal   | 5.464M     | 8.841G     | 0.68     | 23.8     | --> |
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