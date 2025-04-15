# :artificial_satellite: :t-rex: SatDINO: A Deep Dive into Self-Supervised Pretraining for Remote Sensing


This is the official implementation of "SatDINO: A Deep Dive into Self-Supervised Pretraining for Remote Sensing" — a self-supervised learning framework tailored for satellite imagery. SatDINO builds upon the **[DINO](https://github.com/facebookresearch/dino)** framework and adapts it to the unique remote sensing data.

Code is based on official **[DINO](https://github.com/facebookresearch/dino)** implementation.

[ **[Paper]()** ]


## Pretrained models

The models are pretrained on the RGB variant of the fMoW dataset and evaluated across multiple standard remote sensing benchmarks.

| arch      | patch size | params. | GFLOPs | linear | hugging face                                                                          | weights                                                                                           | weights-finetune                                                                                           |
|-----------|------------|---------|--------|--------|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| vit_small | 16         | 21.59   | 8.54   | 72.75  | [strakajk/satdino-vit_small-16](https://huggingface.co/strakajk/satdino-vit_small-16) | [ckp](https://huggingface.co/strakajk/satdino-vit_small-16/resolve/main/satdino-vit_small-16.pth) | [ckp](https://huggingface.co/strakajk/satdino-vit_small-16/resolve/main/satdino-vit_small-16-finetune.pth) |
| vit_small | 8          | 21.37   | 33.56  | 73.53  | [strakajk/satdino-vit_small-8](https://huggingface.co/strakajk/satdino-vit_small-8)   | [ckp](https://huggingface.co/strakajk/satdino-vit_small-8/resolve/main/satdino-vit_small-8.pth)   | [ckp](https://huggingface.co/strakajk/satdino-vit_small-8/resolve/main/satdino-vit_small-8-finetune.pth)   |
| vit_base  | 16         | 85.65   | 33.90  | 73.52  | [strakajk/satdino-vit_base-16](https://huggingface.co/strakajk/satdino-vit_base-16)   | [ckp](https://huggingface.co/strakajk/satdino-vit_base-16/resolve/main/satdino-vit_base-16.pth)   | [ckp](https://huggingface.co/strakajk/satdino-vit_base-16/resolve/main/satdino-vit_base-16-finetune.pth)   |


### Create from HF
You can create model using Hugging Face or directly from the repository.

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("strakajk/satdino-vit_small-16", trust_remote_code=True)
model.eval()

# predict
x = torch.randn(1, 3, 224, 224)
y = model(x)   # out: torch.Size([1, 384])
```

### Create manually
If you are creating model from the repository you can also load classification head trained on fMoW.

```python
import torch
from satdino.vision_transformer_satdino import vit_small, LinearClassifier

checkpoint_path = "checkpoints/satdino-vit_small-16.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# load model
model = vit_small(patch_size=16)
model.load_state_dict(checkpoint['teacher'], strict=True)
model.eval()

# optional: load classification head
head = LinearClassifier(model.embed_dim, 63)
head.load_state_dict(checkpoint['linear_head'], strict=True)
head.eval()

# predict
x = torch.randn(1, 3, 224, 224)
y = model(x)   # out: torch.Size([1, 384])
y = head(y)    # out: torch.Size([1, 63])
```

## Results
| Dataset   | **SatDINO$_8$** | **SatDINO$_{16}$** | **Scale-MAE** | **SatMAE** |
|-----------|-----------------|--------------------|---------------|------------|
| EuroSAT   | **87.72**       | 85.96              | 85.42         | 81.43      |
| RESISC45  | **85.29**       | 82.32              | 79.96         | 65.96      |
| UC Merced | **94.82**       | 93.21              | 84.58         | 78.45      |
| WHU-RS19  | **98.18**       | 97.82              | 89.32         | 86.41      |
| RS-C11    | **96.91**       | 96.61              | 93.03         | 83.96      |
| SIRI-WHU  | **91.82**       | 87.19              | 84.84         | 77.76      |
Average kNN classification accuracy across multiple scales (12.5%, 25%, 50%, and 100%).

---

| **Dataset** | **Small$_{16}$** | **Small$_8$** | **Base**      |
|-------------|------------------|---------------|---------------|
| EuroSAT     | 98.69            | 98.76         | **98.83**     |
| RESISC45    | 95.68            | 95.16         | **96.05**     |
| UC Merced   | 98.33            | **98.81**     | 98.57         |
| WHU-RS19    | **98.54**        | 98.06         | 97.57         |
| RS-C11      | **98.01**        | 96.81         | 96.02         |
| SIRI-WHU    | **98.54**        | 97.08         | 97.08         |
SatDINO fine-tuning classification accuracy.

---

| **Model** | **Backbone**     | **Potsdam 224$^2$** | **Potsdam 512$^2$** | **Vaihingen 224$^2$** | **Vaihingen 512$^2$** | **LoveDA 224$^2$** | **LoveDA 512$^2$** |
|-----------|------------------|---------------------|---------------------|-----------------------|-----------------------|--------------------|--------------------|
| SatMAE    | ViT-Large        | 67.88               | 70.39               | 64,81                 | 69.13                 | 46.28              | 52.28              |
| Scale-MAE | ViT-Large        | 69.74               | **72.21**           | 67.97                 | **71.65**             | **49.37**          | **53.70**          |
| SatDINO   | ViT-Small$_{16}$ | 67.93               | 71.80               | 63.38                 | 68.32                 | 44.77              | 49.65              |
| SatDINO   | ViT-Small$_8$    | **70.71**           | 71.45               | **68.69**             | 67.71                 | 47.53              | 50.20              |
| SatDINO   | ViT-Base         | 67.65               | 71.63               | 64.85                 | 69.37                 | 44.25              | 50.08              |
Semantic segmentation performance across multiple datasets and image scales. All results are reported in terms of mean Intersection over Union (mIoU).



## Environment
```bash
conda create -n satdino python=3.12
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data preparation

### Training data
1. Download RGB version of fMoW dataset https://github.com/fMoW/dataset
2. Preprocess using code https://github.com/fMoW/baseline/blob/master/code/fmowBaseline.py
3. Save only relevant information into csv using: 
```python
from satdino.fmow_prepare import data_to_csv

input_file = "data/fmow/fMoW_processed/working/training_struct.json"
output_file_train = "data/fmow/_tmp/train_split.csv"
output_file_val = "data/fmow/_tmp/val_split.csv"

data_train = data_to_csv(input_file, output_file_train, split_name="train")
data_val = data_to_csv(input_file, output_file_val, split_name="val")
```

### Evaluation: kNN
You can use any image classification dataset using dataset class in `satdino/classification_dataset.py` if the dataset is in following folder structure:
```

dataset/
├─ images/
│  ├─ class_00/
│  │  ├─ image_000.png
│  │  ├─ image_001.png
│  │  ├─ ...
│  ├─ class_01/
│  │  ├─ ...
│  ├─ .../
├─ train.txt
├─ val.txt

```

Usage:
```python
from satdino.classification_dataset import ClassificationDataset

image_folder = "data/eurosat/images"
train_names = "data/eurosat/images/train.txt"
dataset_train = ClassificationDataset(train_names, image_folder)
```

## Train
<details>
    <summary>Train SatDINO model</summary>

```bash
OUTPUT_FOLDER="output/satdino-vit_small-16"
DATASET_ROOT="data"
TRAIN_FILE_PATH="${data}/fmow/train_split.csv"

torchrun satdino/main_satdino.py \
    --arch vit_small \
    --patch_size 16 \
    --teacher_temp 0.07 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --global_crops_scale 0.25 1 \
    --local_crops_scale 0.05 0.25 \
    --local_crops_number 10 \
    --norm_last_layer False \
    --clip_grad 0 \
    --epochs 200 \
    --warmup_epochs 10 \
    --warmup_teacher_temp_epochs 30 \
    --seed 0 \
    --lr 0.001 \
    --min_lr 0.00002 \
    --batch_size_per_gpu 64 \
    --num_workers 8 \
    --saveckp_freq 25 \
    --data_path "${TRAIN_FILE_PATH}" \
    --data_root "${DATASET_ROOT}" \
    --output_folder "${OUTPUT_FOLDER}" \
    --normalization "fmow" \
    --augmentation_type "satdino" \
    --dataset_type "temporal" \
    --temporal_dataset False \
    --gsd_loss "mse" \
    --gsd_weight 0.1 
```
</details>

<details>
    <summary>Train linear head</summary>

```bash
DATASET_ROOT="data"
VAL_FILE_PATH="${data}/fmow/val_split.csv"
TRAIN_FILE_PATH="${data}/fmow/train_split.csv"
CHECKPOINT_PATH="checkpoints/satdino-vit_small-16.pth"
OUTPUT_FOLDER="output/satdino-vit_small-16"

torchrun satdino/eval_linear.py \
  --arch vit_small \
  --patch_size 16 \
  --model_type satdino \
  --normalization fmow \
  --num_workers 8 \
  --epochs 25 \
  --lr 0.00001 \
  --data_root "${DATASET_ROOT}" \
  --val_data_path "${VAL_FILE_PATH}" \
  --train_data_path "${TRAIN_FILE_PATH}" \
  --output_folder "${OUTPUT_FOLDER}" \
  --pretrained_weights "${CHECKPOINT_PATH}" \
  --num_labels 63 \
  --finetune_mode head # or full
```
</details>

## Eval
<details>
    <summary>kNN eval</summary>

```bash
DATASET_ROOT="data"
CHECKPOINT_PATH="checkpoints/satdino-vit_small-16.pth"
OUTPUT_FILE="output"

python satdino/eval_knn.py \
  --dataset_folder  "${DATASET_ROOT}eurosat/images" \
                    "${DATASET_ROOT}resisc45/images" \
                    "${DATASET_ROOT}rs_c11/images" \
                    "${DATASET_ROOT}siri-whu/images" \
                    "${DATASET_ROOT}uc_merced/images" \
                    "${DATASET_ROOT}whu-rs19/images" \
  --pretrained_weights "${CHECKPOINT_PATH}" \
  --model_type satdino \
  --arch vit_small \
  --normalization dataset \
  --scales 1.0 0.5 0.25 0.125 \
  --output_file "${OUTPUT_FILE}"
```
</details>

<details>
    <summary>Linear eval</summary>

```bash
DATASET_ROOT="data"
VAL_FILE_PATH="${data}/fmow/train_split.csv"
CHECKPOINT_PATH="checkpoints/satdino-vit_small-16.pth"
OUTPUT_FOLDER="output/satdino-vit_small-16"

torchrun satdino/eval_linear.py \
  --arch vit_small \
  --patch_size 16 \
  --model_type satdino \
  --normalization fmow \
  --num_workers 8 \
  --data_root "${DATASET_ROOT}" \
  --val_data_path "${VAL_FILE_PATH}" \
  --output_folder "${OUTPUT_FOLDER}" \
  --pretrained_weights "${CHECKPOINT_PATH}" \
  --num_labels 63 \
  --evaluate
```
</details>


## License
This repository is released under the Apache 2.0 license as found in the LICENSE file.


## Citation
If you find this repository useful, please consider citing it:
```
```





















