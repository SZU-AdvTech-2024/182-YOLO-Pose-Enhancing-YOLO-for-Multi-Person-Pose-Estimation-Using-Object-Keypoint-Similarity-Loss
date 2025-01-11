# YOLO-Pose 多人姿态估计模型

这个代码库是论文["**YOLO-Pose: 使用对象关键点相似度损失增强YOLO进行多人姿态估计**"](https://arxiv.org/abs/2204.06806)的官方实现,该论文被CVPR 2022的深度学习高效计算机视觉(ECV)研讨会接收。这个代码库包含基于YOLOv5的人体姿态估计模型。代码从https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose 分支而来。

这个代码库基于YOLOv5训练,并假定已经安装了YOLOv5训练所需的所有依赖。下面是一个示例推理。
<br/> 
<p float="left">
<img width="800" src="./utils/figures/AdobeStock.gif">
</p>     
在COCO验证集上,YOLO-Pose在AP50指标上优于所有其他自下而上的方法,如下图所示:
<br/> 

<p float="left">
<img width="800" src="./utils/figures/AP50_GMACS_val.png">
</p>     

* 下面是与现有的基于关联嵌入的HigherHRNet方法在COCO val2017数据集的拥挤场景图像上的对比示例。

YOLOv5m6-pose的输出             |  HigherHRNetW32的输出  
:-------------------------:|:-------------------------:
<img width="600" src="./utils/figures/000000390555_YP.jpg"> |  <img width="600" src="./utils/figures/000000390555_AE.jpg">

## **数据集准备**
数据集需要准备成YOLO格式,以便数据加载器可以读取关键点和边界框信息。使用了这个[repository](https://github.com/ultralytics/JSON2YOLO)并做了必要的修改来生成所需格式的数据集。
请从[这里](https://drive.google.com/file/d/1irycJwXYXmpIUlBt88BZc2YzH__Ukj6A/view?usp=sharing)下载处理好的标签。建议创建一个新目录coco_kpts并从COCO数据集创建**images**和**annotations**目录的软链接。将**下载的标签**和文件**train2017.txt**和**val2017.txt**放在coco_kpts文件夹中。

预期的目录结构:

```
edgeai-yolov5
│   README.md
│   ...   
│
coco_kpts
│   images
│   annotations
|   labels
│   └─────train2017
│       │       └───
|       |       └───
|       |       '
|       |       .
│       └─val2017
|               └───
|               └───
|               .
|               .
|    train2017.txt
|    val2017.txt
```

## **YOLO-Pose Models and Ckpts**

|Dataset | Model Name                                                                                                                                                                                                                                              |Input Size |GMACS  |AP[0.5:0.95]%| AP50%|Notes |
|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|----------|-------------|------|----- |
|COCO    | [Yolov5s6_pose_640](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_640_60p7_85p3_kpts_head_6x_dwconv_3x3_lr_0p01/weights/last.pt) |640x640    |**10.2**  |   57.5      | 84.3 | [opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_640_60p7_85p3_kpts_head_6x_dwconv_3x3_lr_0p01/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_640_60p7_85p3_kpts_head_6x_dwconv_3x3_lr_0p01/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5s6_960_71p6_93p1/weights/last.pt)|
|COCO    | [Yolov5s6_pose_960](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_960_64p8_87p4_kpts_head_6x_dwconv_3x3_lr_0p01/weights/last.pt) |960x960    |**22.8**  |   63.7      | 87.6 | [opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_960_64p8_87p4_kpts_head_6x_dwconv_3x3_lr_0p01/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5s6_960_64p8_87p4_kpts_head_6x_dwconv_3x3_lr_0p01/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5s6_960_71p6_93p1/weights/last.pt)|
|COCO    | [Yolov5m6_pose_960](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5m6_960_67p8_89p3_kpts_head_6x_dwconv_3x3_lr_0p01/weights/last.pt) |960x960    |**66.3**  |   67.4      | 89.1 | [opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5m6_960_67p8_89p3_kpts_head_6x_dwconv_3x3_lr_0p01/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5m6_960_67p8_89p3_kpts_head_6x_dwconv_3x3_lr_0p01/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5m6_960_74p1_93p6/weights/last.pt)|
|COCO    | [Yolov5l6_pose_960](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5l6_960_69p6_90p1_kpts_head_6x_dwconv_3x3_lr_0p01/weights/last.pt) |960x960    |**145.6** |   69.4      | 90.2 | [opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5l6_960_69p6_90p1_kpts_head_6x_dwconv_3x3_lr_0p01/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/best_models/yolov5l6_960_69p6_90p1_kpts_head_6x_dwconv_3x3_lr_0p01/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5l6_960_74p7_94p0/weights/last.pt)|

## **训练:YOLO-Pose**
通过运行以下命令使用上一节中合适的预训练检查点来训练合适的模型:

```python
python train.py --data coco_kpts.yaml --cfg yolov5s6_kpts.yaml --weights '预训练检查点路径' --batch-size 64 --img 960 --kpt-label
                                      --cfg yolov5m6_kpts.yaml 
                                      --cfg yolov5l6_kpts.yaml 
```

要以640的输入分辨率训练模型,运行以下命令:
```python
python train.py --data coco_kpts.yaml --cfg yolov5s6_kpts.yaml --weights '预训练检查点路径' --batch-size 64 --img 640 --kpt-label
```

## **YOLOv5-ti-lite Based Models and Ckpts**

|Dataset |Model Name                      |Input Size |GMACS  |AP[0.5:0.95]%| AP50%|Notes |
|--------|------------------------------- |-----------|----------|-------------|------|----- |
|COCO    |[Yolov5s6_pose_640_ti_lite](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_640_ti_lite_54p9_82p2/weights/last.pt)     |640x640    |**8.6**  |  54.9      | 82.2 |[opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_640_ti_lite_54p9_82p2/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_640_ti_lite_54p9_82p2/hyp.yaml), [pretrained_weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/other/person_detector/yolov5s6_ti_lite_person_64p8_90p2/weights/last.pt)|
|COCO    |[Yolov5s6_pose_960_ti_lite](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_960_ti_lite_59p7_85p6/weights/last.pt)     |960x960    |**19.3** |  59.7      | 85.6 |[opt.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yolov5/yolov5s6_960_ti_lite_59p7_85p6/opt.yaml), [hyp.yaml](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_02_00_11/edgeai-yolov5/pretrained_models/checkpoints/keypoint/coco/edgeai-yo