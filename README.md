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
+ **examples** 样例（暂未实现）
+ **log** TensorBoard日志存储路径
+ **models** 模型文件夹
+ **output** 验证以及离线推理输出文件夹
  + **inference** 离线推理结果输出文件夹
  + `pred.mat` 验证集预测结果
  + `val_*_gt.jpg` 验证集ground truth真实标签
  + `val_*_hm_gt.jpg` 验证集热图真实标签
  + `val_*_pred.jpg` 验证集预测结果
  + `val_*_hm_pred.jpg` 验证集热图预测结果
+ **utils** 工具类代码文件夹
  + `config.py` 项目配置文件，配置说明见`config.md`
  + `data_loader.py` 数据集加载模块
  + `dataset_crawler.py` 数据集链接爬取模块，可独立运行
  + `evaluate.py` 数据集评分模块
  + `loss.py` 定义了`JointsMSELoss`用于计算姿态估计的损失
  + `mpii.py` 提供MPII数据集的dataset接口
  + `optimizer.py` 定义了网络所用的优化器与scheduler(中文名没想到叫啥)
  + `setup.py` 项目初始化模块
  + `tools.py` 其余杂项的模块，如推理、存储参数、模型摘要等
  + `transforms.py` 数据变换模块，对数据进行数据增强处理
  + `validate.py` 验证集损失计算模块
  + `vis.py` 数据可视化模块    
+ `main.py` 项目运行入口
 
# 规划
A light network to recognize fallen person. 
## Refrence
+ [TCFormer](https://arxiv.org/pdf/2204.08680.pdf)
+ [Simple BaseLine](https://arxiv.org/pdf/1804.06208.pdf)
+ [HRNet](https://arxiv.org/pdf/1902.09212.pdf)
+ ...
## TODO

- [x] Dataset
    - [x] MPII
    - [x] Fallen
    
- [x] **TCFormer**
    - [x] Code
    - [x] Result(Possible)
      - Acc: 0.87
      - fps: 1.2 on cpu.
- [x] **MobileNetV3 with SimpleHead**
    - [x] Code
    - [x] Result(Possible)
      - Acc: 0.74
      - fps: 35 on cpu.
- [ ] **HRNet**
    - [ ] Code
    - [ ] Result(Possible)
- [ ] **VIPNAS**
    - [ ] Code
    - [ ] Result(Possible)
    
 - [ ] **MHFormer**
    - [ ] Code
    - [ ] Result(Possible)
   
  - [ ] **YOLO**
    - [ ] Code
    - [ ] Result(Possible)
