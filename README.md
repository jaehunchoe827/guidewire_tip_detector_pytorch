# Guidewire detection network.
This repo is implemeted based on jahongir7174's [YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)

# parameters
1. yolo model size
2. head # units, # conv layers

## TODO
0. consider using trainable upsamplers

## Target
1%_win_acc = 0.6191, and 2%_win_acc = 0.8231

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
