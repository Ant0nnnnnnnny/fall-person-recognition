# config.py 配置文件说明

## Project 部分
+ `model_name` 模型名称。通过指定模型名称来加载不同的模型文件。
+ `seed` 随机种子。
+ `dataset_root` 数据集文件夹名称。
+ `ckpg_dir` 模型存放配置文件夹名称。
+ `inference_dir` 离线推理结果文件夹名称。

## Data 部分
+ `img_shape` 输入的图像大小。
+ `prefetch` 数据集提前缓存的大小。
+ `num_workers` dataloader并行载入的线程数量。
+ `batch_size` 训练集的batch大小。
+ `val_batch_size` 验证集的batch大小。
+ `test_batch_size` 测试集batch大小。
+ `num_joints_half_body` 半身的关键点数量，如果大于这个数量则对数据进行数据增强处理。即将完整的人体拆分为半身图进行训练。
+ `flip` 是否对数据集进行翻转数据增强处理。
+ `rotation_factor` 数据增强旋转角度，逆时针。
+ `scale_factor` 数据集放缩大小，防止截断人体。实际放缩大小为 `1 + scale_factor`。
## Optimizer 部分
+ `learning_rate` 学习率。
+ `scheduler_factor`  `ReduceLROnPlateau` 中`learning rate`递减速率。
+ `scheduler_patience` `ReduceLROnPlateau` 中`learning rate`递减阈值。
+ `scheduler_min_lr`  `ReduceLROnPlateau` 中`learning rate`的最小值。
## Model Configs 部分
+ `config_dir` 各种模型config配置文件夹。

## Train相关
+ `max_epochs` 最大epoch。
+ `print_steps` 设定打印输出的step补偿。
+ `log_dir` TensorBoard的log保存文件夹。
+ `auto_resume` 是否自动续训。
+ `output_dir` 推理以及离线验证文件夹名称。
+ `debug_dir` 模型训练结果可视化文件。
## Debug相关
+ `post_processing` 是否打印评估结果。
+ `debug_mode` 是否为调试模式。
+ `save_batch_image_gt` 是否保存每个batch中的ground truth。
+ `save_batch_image_pred` 是否保存每个batch中的预测结果。
+ `save_batch_heatmap_gt` 是否保存每个batch中heatmap的ground truth。
+ `save_batch_heatmap_pred` 是否保存每个batch中heatmap的预测结果。
