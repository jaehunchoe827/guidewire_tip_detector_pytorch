# Guidewire detection network.
This repo is implemeted based on jahongir7174's [YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)

# parameters
0. consider using trainable upsamplers
1. yolo model size
2. head # units, # conv layers

## Target
1%_win_acc = 0.6191, and 2%_win_acc = 0.8231

## best so far : using bilinear upsampling
(guidewire_detector) jaehun@jh22:~/workspace/guidewire_tip_detector_pytorch$ python3 -m engine.test_unfreeze --train --config wider_head.yaml
Start training...
config loaded. number of keys: 6
Loading weights from /home/jaehun/workspace/guidewire_tip_detector_pytorch/weights/yolo_v11_s.pt
Total parameters: 10,051,266
Trainable parameters: 10,051,266
Epoch 1/30 - val_loss_total: 0.001040, val_acc5: 0.57946, val_acc1: 0.47203                                             
Epoch 2/30 - val_loss_total: 0.000727, val_acc5: 0.76449, val_acc1: 0.65756                                             
Epoch 3/30 - val_loss_total: 0.000749, val_acc5: 0.72626, val_acc1: 0.63038                                             
Epoch 4/30 - val_loss_total: 0.000748, val_acc5: 0.73781, val_acc1: 0.63432                                             
Backbone unfreezed at epoch 5
Epoch 5/30 - val_loss_total: 0.001147, val_acc5: 0.61976, val_acc1: 0.52933                                             
Epoch 6/30 - val_loss_total: 0.000698, val_acc5: 0.77854, val_acc1: 0.67556                                             
Epoch 7/30 - val_loss_total: 0.000830, val_acc5: 0.65548, val_acc1: 0.55149                                             
Epoch 8/30 - val_loss_total: 0.000696, val_acc5: 0.80020, val_acc1: 0.69256                                             
Epoch 9/30 - val_loss_total: 0.000641, val_acc5: 0.81411, val_acc1: 0.71264    


### Installation
```
conda create -n guidewire_detector python=3.10.10
conda activate guidewire_detector
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python PyYAML tqdm thop seaborn pandas
```
After installation, you must add the following line to your ~/.bashrc
```
export CUBLAS_WORKSPACE_CONFIG=:16:8
```
### Train
* **For reproducibility**, you must uncomment the below line in utils.util.py
    ```
    # torch.use_deterministic_algorithms(True)
    ```
* Configure your dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Results

| Version | Epochs | Box mAP |                                                                              Download |
|:-------:|:------:|--------:|--------------------------------------------------------------------------------------:|
|  v11_n  |  600   |    38.6 |                                                            [Model](./weights/best.pt) |
| v11_n*  |   -    |    39.2 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_n.pt) |
| v11_s*  |   -    |    46.5 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_s.pt) |
| v11_m*  |   -    |    51.2 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_m.pt) |
| v11_l*  |   -    |    53.0 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_l.pt) |
| v11_x*  |   -    |    54.3 | [Model](https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_x.pt) |

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.551
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.196
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.361
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.777
```

* `*` means that it is from original repository, see reference
* In the official YOLOv11 code, mask annotation information is used, which leads to higher performance


### Dataset structure
    ├── datasets
        ├── images
            ├── guidewire
                ├── 01_04
                    ├── Image
                        ├── 1234.jpg
                        ├── 2222.jpg (not all images have corresponding labels.)
                        ├── 3333.jpg
                    ├── Labels
                        ├── 1234.jpg
                        ├── 3333.jpg
                ├── 05_09
                    ├── Image
                        ├── ...
                    ├── Labels
                        ├── ...
                ├── ...

#### Reference
* https://github.com/jahongir7174/YOLOv11-pt
* https://github.com/ultralytics/ultralytics
