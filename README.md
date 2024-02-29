SCOAD: Single-frame Click Supervision for Online Action Detection
------
This repository is the official implementation of SCOAD. In this work, we propose a weakly supervised online action detection method with click labels.
![在这里插入图片描述](https://github.com/zstarN70/SCOAD/blob/main/framework.png)


-----
## Environment
To install requirements:
```
conda env create -n env_name -f environment.yaml
```
Before running the code, please activate this conda environment.
## Data Preparation
a. Download pre-extracted features from [here](https://github.com/sujoyp/wtalc-pytorch#data).  
b. The manual annotation of the background from [here](https://github.com/VividLe/BackTAL/tree/main/data/THUMOS14/human_anns).  
c. The manual annotation of the action from [here](https://github.com/Flowerfan/SF-Net/tree/master/data/Thumos14-Annotations/single_frames).  
d. We process the manual annotations into pickle files:  
```
python scripts/thumos_click_deal_action.py
python scripts/thumos_click_deal_back.py
```

Please ensure the data structure is as below

```
data
└── Thumos14
    ├── signal_anno
    │   ├── human
    │   │   ├── action
    │   │   │   ├── THUMOS1.txt
    │   │   │   └── ...
    │   │   └── back 
    │   │       └── THUMOS14_Background-Click-Annotation_A1.txt
    │   └── seed
    │       ├── random_seed_action
    │       └── random_seed_back
    ├── feature
    │   └── Thumos14reduced-I3D-JOINTFeatures.npy
    └── Thumos14reduced-Annotations
        ├── Ambiguous_test.txt
        ├── classlist.npy
        ├── duration.npy
        └── ...
```

## Train
a. Config
Adjust configurations according to your machine.

```
./misc/config.py
```
b. Train
```
python main.py
```
## Inference
a. You can download pre-trained models from [here](https://drive.google.com/drive/folders/1aBIefa_MqJF_rs_wF75h26zRyE4pivlA?usp=sharing), and put the weight file in the folder checkpoint.
```
python eval.py --pretrained_ckpt MODEL_NAME
```

## Acknowledgement
We referenced [WOAD](https://github.com/salesforce/woad-pytorch.git) for the code.
