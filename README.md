# fallen-person-recognition
<div align="center">

English | [简体中文](README_CN.md)

</div>

## NEWS
 2023.8 —— **Real-time Skeleton-Based Fall Detection** is available now.**F1 Score:93.6%**.

 2023.8 —— **Skeleton-Based Fall Detection dataset** are available new, which is reprocessed from [UR Fall Dataset](http://fenix.ur.edu.pl/~mkepski/ds/uf.html), containing total 6872 samples.[Google Drive](https://drive.google.com/drive/folders/1Kz85dnCAErvlML9H-7WAhEzKbbWLHPKb?usp=drive_link)

 2023.8 —— Real-time multi-person skeleton-based action recognition are available now. **31FPS** on Apple M1Pro(CPU)。

 2023.8 —— Real-time multi-person pose estimation are available now. **33FPS** on Apple M1Pro(CPU)。

 2023.7 —— **Skeleton-Based Action recognition dataset has been added to project**，Which contains over 110000 samples. [Google Drive](https://drive.google.com/drive/folders/1-n0jYog_vLufOdzq5lYgvuI1q_ulrpD8?usp=drive_link)

 2023.7 —— **Multi-pose estimation** are available now.

 2023.2 —— **Real-time single person pose estimation** is available. 

## TODO
- [x] Retrained on COCO dataset.（**47FPS**）.
- [x] Real-time human body detector based on [PicoDet](https://arxiv.org/pdf/2111.00902.pdf)（**73FPS**）.
- [x] Pose tracking based on [ByteTrack](https://arxiv.org/pdf/2110.06864.pdf).
  - [x] Bounding boxes tracking.
  - [x] Skeletons tracking.
- [x] Action recognition model based on deep learning.
  - [x] Dataset（NTU-120）。
  - [x] experiment on related methods.
  - [x] Constructing on light model.
  - [x] Improve model performance.
- [ ] Improve model performance.
  - [x] Skeleton noise filter in video inference.
  - [ ] Improve PicoDet.
  - [ ] Improve Pipeline.
  - [ ] Improve Tracker.
- [x] Release python deployed version.
- [ ] Release C++ deployed version.
- [ ] Computing parallel on servers.

## Usage
```{bash}
python video.py --video_path YOUR_VIDEO_PATH [-- enable_filter True]
```

## Docs
Config→[Config](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/docs/config.md)


Projects files info: [Files](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/docs/structure.md)

## Download
### Skeleton-Based Action Recognition Dataset
Link:[Google Drive](https://drive.google.com/drive/folders/1-n0jYog_vLufOdzq5lYgvuI1q_ulrpD8?usp=drive_link)(No limit)  [Baidu Yun](https://pan.baidu.com/s/1Mw040S7RUPSiRFxxCGgxZA?pwd=p7sc)
### Skeleton-Based Fall Detection Dataset
Link:[Google Drive](https://drive.google.com/drive/folders/1Kz85dnCAErvlML9H-7WAhEzKbbWLHPKb?usp=drive_link)(No limit) [Baidu Yun](https://pan.baidu.com/s/1Usden9TvZ1Gw2FDZTdlRdw?pwd=4v7n)

## Results

### Skeleton-based action recognition model performances

| Model                                                 | Accuracy in experiment(%) | Accuracy in paper(%)                               | latency(ms) | Params(M) |
| ----------------------------------------------------- | ------------------------- | -------------------------------------------------- | ----------- | --------- |
| [ST-GCN](https://arxiv.org/pdf/1801.07455v2.pdf)      | 85.8                      | [88.8(3D)](https://arxiv.org/pdf/1801.07455v2.pdf) | 79.2  ± 4.4 | 3.095     |
| [SGN](https://arxiv.org/pdf/1904.01189.pdf)           | 74.6                      | [79.2(3D)](https://arxiv.org/pdf/1904.01189.pdf)   | 6.55 ± 0.47 | 0.721     |
| SGN+（Ours）                                          | 84.3                      | -                                                  | 12.4 ± 0.35 | 0.662     |
| **TODO** [STID](https://arxiv.org/pdf/2208.05233.pdf) | -                         | -                                                  | -           | -         |

### Skeleton-based Fall detection results

![Gif1](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/Clip1.gif)

![Gif2](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/Clip2.gif)

![Gif3](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/Clip3.gif)


### SGN+
![Figure](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/SGN-accuracy.png)

### Model's performances in pipeline 

| Model                                             | FLOPS(G) | Params(M) | Latency(ms/frame)`*` | Info                                                                                        |
| ------------------------------------------------- | -------- | --------- | -------------------- | ------------------------------------------------------------------------------------------- |
| MFNet                                             | 0.67     | 4.10      | 9.2/Person           | Pose estimation model.                                                                      |
| [PicoDet](https://arxiv.org/pdf/2111.00902.pdf)   | 1.18     | 0.97      | 13.7                 | Human detection model.                                                                      |  |
| [ByteTrack](https://arxiv.org/pdf/2110.06864.pdf) | -        | -         | 7.3                  | Human tracking model.                                                                       |
| [SGN+](https://arxiv.org/pdf/1904.01189.pdf)      | 2.73     | 0.662     | 12.4                 | Action recognition model.                                                                   |
| Total                                             | 4.58     | 6.09      | 42.6                 | [Video](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/video.mov) |

`*`Info: Evaluating on M1Pro(8+2)。

![Result](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/ST-GCN_Skeleton.jpg)

### Multi-person: Cover
![足球](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/multi-pose-estimation.png)

 Detector: [PicoDet](https://arxiv.org/pdf/2111.00902.pdf)
 Pose estimator: **MFNet** 

### Siting

![Sitting](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/sit-pose-estimation.png)
 Detector: [PicoDet](https://arxiv.org/pdf/2111.00902.pdf)
 Pose estimator: **MFNet** 

### Standing

![Standing](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/stand-pose-estimation.png)
 Detector: [PicoDet](https://arxiv.org/pdf/2111.00902.pdf)
 Pose estimator: **MFNet** 

### Wide
![Wide](https://github.com/qhtLucifer/fallen-person-recognize/blob/main/examples/wide_angle1.jpg)
 Detector: [PicoDet](https://arxiv.org/pdf/2111.00902.pdf)
 Pose estimator:  **MFNet** 

## Plan
### Stage 1
Build a real-time pose estimation model**Done**
  
Target: **AP: 0.80+**,**FPS: 18+** 
Result: **AP: 0.93**,**FPS: 47+** 
### Stage 2
Build a light skeleton-based fallen recognition model based on pose estimator in stage 1.
  
Target: **Accuracy: 0.95+**, **CPU-REALTIME**

### Stage 3
Deploy models based on `micro-python`,`Django`and `Flutter`. `micro-python` is used to capture data on edge devices and upload to server.`Django` or any server framework is used to build backend，which is responsible for infer frames and send results to mobile devices.`Flutter` is used to build an app to receive results computed by server, such as whether the target is falling and so on。
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