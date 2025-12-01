# Guidewire detection network.
This repo is implemeted based on jahongir7174's [YOLOv11-pt](https://github.com/jahongir7174/YOLOv11-pt)

### Installation
```
conda create -n gwtd python=3.10
conda activate gwtd
conda install nvidia/label/cuda-12.8.1::cuda-toolkit
conda install nvidia/label/cudnn-9.7.1::cudnn
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
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
```
python3 -m engine.main --train --config ver0_default.yaml
```

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
