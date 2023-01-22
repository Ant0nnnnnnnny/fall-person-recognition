# fallen-person-recognize
## 项目目录
+ **checkpoints** 模型参数存放文件夹
  + `checkpoint.pth` 当前训练的最新一轮的模型参数
  + `model_best.pth` 历史训练的得分最高的模型参数
+ **dataset** 数据集文件夹，包括跌倒数据集，姿态估计数据集
  + **FallenData** 跌倒数据集文件夹
    + **Annotation** 标签数据
    + **zip** 图像数据
  + **PoseData** 姿态估计数据集文件夹
    + **Annotation** 标签数据
    + **zip** 图像数据
+ **debug** 每一轮的训练结果可视化图片存储路径
  + `*_gt.jpg` ground truth 真实标签
  + `*_hm_gt.jpg` 热图真实标签
  + `*_ht_pred.jpg` 热图预测结果
  + `*_pred.jpg` 真实标签预测结果
+ **examples** 测试样例，用于推理测试所需的图片、视频等
+ **log** TensorBoard日志存储路径
+ **models** 模型文件夹
+ **output** 验证以及离线推理输出文件夹
  + **inference** 离线推理结果输出文件夹
  + `pred.mat` 验证集预测结果
  + `val_*_gt.jpg` 验证集ground truth真实标签
  + `val_*_hm_gt.jpg` 验证集热图ground truth
  + `val_*_pred.jpg` 验证集预测结果
  + `val_*_hm_pred.jpg` 验证集热图预测结果
+ **utils** 工具类代码文件夹
  + `config.py` 项目配置文件，配置说明见`config.md`
  + `data_loader.py` 数据集加载模块
  + `dataset_crawler.py` 数据集链接爬取模块，可独立运行
  + `evaluate.py` 数据集评分模块
  + `loss.py` 定义了`JointsMSELoss`用于计算姿态估计的损失
  + `mpii.py` 提供MPII数据集的dataset接口
  + `optimizer.py` 定义了网络所用的优化器与scheduler
  + `setup.py` 项目初始化模块
  + `tools.py` 其余杂项的模块，如推理、存储参数、模型摘要等
  + `transforms.py` 数据变换模块，对数据进行数据增强处理
  + `validate.py` 验证集损失计算模块
  + `vis.py` 数据可视化模块    
+ `main.py` 项目运行入口

# 当前模型对比
| 模型名称           | Params     | MACs       | AP       | FPS      |
| ------------------ | ---------- | ---------- | -------- | -------- |
| MobileNetV3-normal | **5.285M** | **3.814G** | 0.65     | **28.5** |
| MobileNetV3-large  | 10.871M    | 4.053G     | 0.76     | 22.7     |
| TCFormer           | 25.624M    | 6.535G     | **0.82** | 9.4      |
| MFNet-normal       | 5.464M     | 8.841G     | 0.68     | 27.4     |
# 规划
## 阶段一
完成对姿态估计模型的建立、训练，确定最终落地使用的姿态估计方案。
  
要求：**AP: 0.80+**,**FPS: 18+** 
## 阶段二
基于阶段一的姿态估计方案，使用姿态估计数据，建立数学模型或轻量级神经网络模型，实现跌倒识别。
  
要求：**Accuracy: 0.95+**, **CPU-REALTIME**

## 阶段三
基于`micro-python`,`Django`,`Flutter`实现模型落地部署。其中，`micro-python`用于在边缘设备记录图像信息，并上传至服务器;`Django`用于后端搭建，基于`pytorch`实现图像推理，并得到识别结果;`Flutter`用于搭建跨平台的App,用于向用户展示识别结果信息（如是否跌倒、姿态信息等）。
## Reference
+ [TCFormer](https://arxiv.org/pdf/2204.08680.pdf)
+ [Simple BaseLine](https://arxiv.org/pdf/1804.06208.pdf)
+ [HRNet](https://arxiv.org/pdf/1902.09212.pdf)
+ [MobileNetV3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)
+ [YOLOV7](https://arxiv.org/pdf/2207.02696.pdf)
+ [YOLOV5](https://github.com/ultralytics/yolov5)