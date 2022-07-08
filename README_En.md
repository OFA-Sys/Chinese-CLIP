[**中文说明**](README.md) | [**English**](README_En.md)

<p align="center">
    <br>
    <img src="assets/Chinese_CLIP_logo_tp.svg" width="400" />
    <br>
<p>
<p align="center">
    <a href="https://opensource.org/licenses/MIT">
        <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
</p>

This is the Chinese version of CLIP. We use a large-scale internal Chinese image-text pair dataset (~200M) to train the model, and we hope that it can help users to achieve cross-modal retrieval and image representation generation for Chinese data. This repo is based on <b>[open_clip project](https://github.com/mlfoundations/open_clip)</b>. We have made some optimization for better performance on Chinese data, and we provide the details in the following. 
<br><br>

## Installation Requirements
To start with this project, make sure that your environment meets the requirements below:

* python >= 3.6.4
* pytorch >= 1.7.1 (with torchvision)
* CUDA Version >= 10.1

Run the following command to install required packages.

```bash
pip install -r requirements.txt
```
<br><br>

## Results
We conducted zero-shot inference and finetuning experiments on MUGE Retrieval, Flickr30K-CN and COCO-CN. Results are shown below:
52.1	76.7	84.4	71.1

**MUGE Text-to-Image Retrieval**:
<table border="1" width="100%">
    <tr align="center">
        <th>Setup</th><th colspan="4">Zero-shot</th><th colspan="4">Finetune</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td>
    </tr>
	<tr align="center">
        <td>Wukong<sub>ViT-B</sub></td><td>33.4</td><td>59.3</td><td>69.7</td><td>54.1</td><td>39.2</td><td>66.9</td><td>77.4</td><td>61.2</td>
    </tr>
	<tr align="center">
        <td>R2D2<sub>ViT-B</sub></td><td>-</td><td>-</td><td>-</td><td>-</td><td>47.4</td><td>75.1</td><td>83.5</td><td>68.7</td>
    </tr>
	<tr align="center">
        <td>CN-CLIP<sub>ViT-B</sub></td><td>52.1</td><td>76.7</td><td>84.4</td><td>71.1</td><td>56.8</td><td>82.4</td><td>89.3</td><td>76.2</td>
    </tr>
</table>
<br>

**Flickr30K-CN Retrieval**:
<table border="1" width="100%">
	<tr align="center">
        <th>Task</th><th colspan="6">Text-to-Image</th><th colspan="6">Image-to-Text</th>
    </tr>
    <tr align="center">
        <th>Setup</th><th colspan="3">Zero-shot</th><th colspan="3">Finetune</th><th colspan="3">Zero-shot</th><th colspan="3">Finetune</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td>
    </tr>
	<tr align="center">
        <td>Wukong<sub>ViT-B</sub></td><td>45.7</td><td>73.8</td><td>82.2</td></td><td>67.6</td><td>89.6</td><td>94.2</td><td>66.2</td><td>88.7</td><td>94.3</td></td><td>83.9</td><td>97.6</td><td>99.0</td>
    </tr>
	<tr align="center">
        <td>R2D2<sub>ViT-B</sub></td><td>-</td><td>-</td><td>-</td><td>78.3</td><td>94.6</td><td>97.0</td></td><td>-</td><td>-</td><td>-</td><td>92.6</td><td>99.1</td><td>99.8</td>
    </tr>
	<tr align="center">
        <td>CN-CLIP<sub>ViT-B</sub></td><td>57.0</td><td>82.8</td><td>89.6</td><td>72.9</td><td>92.8</td><td>96.2</td><td>71.0</td><td>90.6</td><td>95.3</td><td>88.6</td><td>98.0</td><td>99.3</td>
    </tr>
</table>
<br>

**COCO-CN Retrieval**:
<table border="1" width="100%">
	<tr align="center">
        <th>Task</th><th colspan="6">Text-to-Image</th><th colspan="6">Image-to-Text</th>
    </tr>
    <tr align="center">
        <th>Setup</th><th colspan="3">Zero-shot</th><th colspan="3">Finetune</th><th colspan="3">Zero-shot</th><th colspan="3">Finetune</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td>
    </tr>
	<tr align="center">
        <td>Wukong<sub>ViT-B</sub></td><td>49.2</td><td>79.4</td><td>87.9</td></td><td>67.0</td><td>91.4</td><td>96.7</td><td>48.3</td><td>77.8</td><td>88.8</td></td><td>65.8</td><td>90.3</td><td>96.6</td>
    </tr>
	<tr align="center">
        <td>R2D2<sub>ViT-B</sub></td><td>-</td><td>-</td><td>-</td><td>75.1</td><td>94.2</td><td>98.1</td></td><td>-</td><td>-</td><td>-</td><td>76.1</td><td>95.3</td><td>98.5</td>
    </tr>
	<tr align="center">
        <td>CN-CLIP<sub>ViT-B</sub></td><td>58.8</td><td>85.3</td><td>93.3</td><td>73.6</td><td>94.9</td><td>97.8</td><td>54.7</td><td>84.7</td><td>92.3</td><td>73.6</td><td>94.8</td><td>98.1</td>
    </tr>
</table>
<br><br>

## Getting Started

### Code Organization

After cloning this project, please create a new directory ```${DATAPATH}``` for datasets, checkpoints and logs。A recommended workspace structure is demonstrated below：

```
Chinese-CLIP/
├── run_scripts/
│   ├── muge_finetune_vit-b-16_rbt-base.sh
│   ├── flickr30k_finetune_vit-b-16_rbt-base.sh
│   └── ...           # more scripts for finetuning and evaluation...
└── src/
    ├── clip/
    ├── eval/
    ├── preprocess/
    └── training/

${DATAPATH}
├── pretrained_weights/
├── experiments/
└── datasets/
    ├── MUGE/
    ├── flickr30k-cn/
    └── .../          # more datasets...
```

### Preparation
We provide links for the downloading of pretrained checkpoints, as well as the data preprocessing procedures for finetuning. 

#### Pretrained Checkpoints

Download the ViT-B checkpoint ([Link](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt)). We recommend putting the checkpoint in `${DATAPATH}/pretrained_weights/`. 

#### Data Preprocessing

We advise to organize the data in the following way to ensure the efficiency of accessing and processing data:

```
${DATAPATH}
└── datasets/
    └── ${dataset_name}/
        ├── train_imgs.tsv      # image id & image content
        ├── train_texts.jsonl   # text id & text content, with list of paired image ids
        ├── valid_imgs.tsv
        ├── valid_texts.jsonl
        ├── test_imgs.tsv
        └── test_texts.jsonl
```
where `${dataset_name}` refers to the name of dataset (e.g., MUGE).

To ensure the efficiency of processing data, we did not store images with small files, but instead we encode them to base64 strings and store them in `${split}_imgs.tsv`. Each line represents an image, where there are id (int) and base64 string, split by `\t`, as shown below:  
```
1000002	/9j/4AAQSkZJ...YQj7314oA//2Q==
```

Transforming image files to base64 strings is simple. Run the following code:
```python
from PIL import Image
from io import BytesIO
import base64

img = Image.open(file_name) # path to file
img_buffer = BytesIO()
img.save(img_buffer, format=img.format)
byte_data = img_buffer.getvalue()
base64_str = base64.b64encode(byte_data) # bytes
base64_str = base64_str.decode("utf-8") # str
```

Texts and image-text pairing relations are stored in `${split}_texts.jsonl`, where each line is a json as shown below:

```
{"text_id": 8428, "text": "高级感托特包斜挎", "image_ids": [1076345, 517602]}
```

Finally, we need to serialize tsv and jsonl and transform them to LMDB files, which is easy for random access during training.
```
python src/preprocess/build_lmdb_dataset.py \
    --data_dir ${DATAPATH}/datasets/${dataset_name}
    --splits train,valid,test
```
For example, for the MUGE dataset, we name `${dataset_name}` to MUGE. `--splits` refers to dataset splits，split by commas without space. After that, there will be LMDB files in the directory.
```
${DATAPATH}
└── datasets/
    └── ${dataset_name}/
        └── lmdb/
            ├── train
            │   ├── imgs
            │   └── pairs
            ├── valid
            └── test
```

### Finetuning

We introduce the procedures of training for users to learn about the details of the model. We finetune with the pretrained Chinese CLIP. For MUGE and Flickr30K-CN, we provide scripts `run_scripts/muge_finetune_vit-b-16_rbt-base.sh` and `run_scripts/flickr30k_finetune_vit-b-16_rbt-base.sh`. The scripts support single-worker and distributed training. Before running, follow the instructions at the beggining of the scripts and fill in your configuration for distributed training. Then run the scripts to start your training. Logs and checkpoints will be saved at your specified paths. 

```bash
cd Chinese-CLIP/
bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}
```

The configuration for training includes:

+ Distributed training
  + `WORKER_CNT`: the number of machines.
  + `GPUS_PER_NODE`: the number of GPUS on each machine.
+ Data for training/validation
  + `train-data`: directory of training data. Follow the procedures above the create LMDB files.
  + `val-data`: directory of validation data.
  + `num-workers`: the number of workers for dataloader.
+ Training hyper-params
  + `vision-model`: specified visual backbones. Select from `["ViT-B-32", "ViT-B-16", "ViT-L-14"]`.
  + `text-model`: specified language backbones. Select from `["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese"]`.
  + `context-length`: sequence length for text inputs.
  + `warmup`: steps for warmup.
  + `batch-size`: batch size for a worker (make sure that the number of training samples larger than `batch-size * GPUs`).
  + `lr`: learning rate.
  + `wd`: weight decay.
  + `max-steps`: training steps. Also you can set `max-epochs` to set the number of training epochs.
  + `freeze-vision`: whether to freeze the visual backbone. 
  + `use-augment`: whether to use [AutoAugment](https://arxiv.org/abs/1805.09501) for data augmentation. 
  + `valid-batch-size`: validation batch size for a worker (make sure that the number of validation samples larger than `valid-batch-size * GPUs`).
  + `valid-step-interval` and `valid-epoch-interval`: validation step / epoch frequency, if set to -1 then validation will be disabled during finetuning.
+ Ouputs
  + `name`: specified output path. Hyperparameter logs, training logs, and checkpoints will be saved at `${DATAPATH}/experiments/${name}/`.
  + `save-step-frequency` and `save-epoch-frequency`: the intervals for saving checkpoints.
  + `report-training-batch-acc`: whether to report the in-batch image-to-text and text-to-image retrieval accuracy. 
+ Checkpoints
  + `resume`: the checkpoint path for weights to restore. In the provided example script, the path refers to the pretrained checkpoint path. Users can change to your own checkpoint path.
  + `reset-data-offset`: whether to restore training at the data breakpoint.
  + `reset-optimizer`: whether to restore the optimizer state。

After training, the log will be saved at `${DATAPATH}/experiments/${name}/out_${timestamp}.log`. Example of log is shown below:
```
2022-06-16,10:58:27 | INFO | Rank 0 | Global Steps: 1/735 | Train Epoch: 1 [1024/250880 (0%)] | Loss: 2.171807 | Image2Text Acc: 49.41 | Text2Image Acc: 52.54 | Data Time: 5.167s | Batch Time: 15.647s | LR: 0.000000 | logit_scale: 4.605 | Global Batch Size: 1024
```
The example of validation log is shown below:
```
2022-06-16,11:06:00 | INFO | Rank 0 | Validation Result (epoch 1 @ 150 steps) | Valid Loss: 0.503617 | Image2Text Acc: 84.76 | Text2Image Acc: 84.37 | logit_scale: 4.605 | Valid Batch Size: 128
```

**Attention**: The convergence and stability of contrastive learning is highly relevant to the total batch size. If you use a smaller batch size, (in comparison with the default 128 per-GPU \* 8 GPU), we advise you to use a smaller learning rat. We recommend using more GPUs and larger batch size for better performance. 

### Inference and Evaluation

We provide procedures for representation generation and cross-modal retrieval, as demonstrated below:

#### Image/Text Representation Generation

By now the code supports representation generation with a single worker. Follow the commands below:
```bash
cd Chinese-CLIP/
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/src

split=valid # validation / test set
resume=${DATAPATH}/pretrained_weights/clip_vit-b-16_roberta-base.pt

python -u src/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=24 \
    --resume=${resume} \
    --vision-model=ViT-B-16 \
    --text-model=RoBERTa-wwm-ext-base-chinese
```

By default, the representations are stored at `${DATAPATH}/datasets/${dataset_name}`. Specifically, the image representations are stored at `${split}_imgs.img_feat.jsonl`. Each line stores a json of image representation, as shown below:
```
{"image_id": 1000002, "feature": [0.0198, ..., -0.017, 0.0248]}
```
Text representations are stored at `${split}_texts.txt_feat.jsonl`，as shown below:
```
{"text_id": 248816, "feature": [0.1314, ..., 0.0018, -0.0002]}
```

#### KNN Retrieval

For small-scale retrieval datasets, we provide a simple implementation of KNN retrieval, to facilitate the retrieval of top-k results in cross-modal retrieval. 

For text-to-image retrieval, run the commands below:
```bash
cd Chinese-CLIP/
split=valid # validation / test splits
python -u src/eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl"
```
Results are stored at specified jsonl files. Each line consists of top-k image ids for a text query, as shown below:
```json
{"text_id": 153915, "image_ids": [5791244, 1009692167, 7454547004, 3564007203, 38130571, 2525270674, 2195419145, 2503091968, 4966265765, 3690431163]}
```

For image-to-text retrieval, run the commands below：
```bash
split=valid # validation / test splits
python -u src/eval/make_topk_predictions_tr.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl"
```
Results are stored at specified jsonl files. Each line consists of top-k text ids for an image query, as shown below:
```json
{"image_id": 977856234, "text_ids": [156914, 157914, 158914, 155914, 156179, 158907, 157179, 154179, 154914, 154723]}
```

#### Recall Metric

We provide scripts for computing the Recall@1/5/10 and mean recall (the mean of Recall@1/5/10). Run the commands to get the scores:

For text-to-image retrieval, run the commands below:
```bash
split=valid # validation / test splits
python src/eval/evaluation.py \
        ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl \
        ${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl \
        output.json
cat output.json
```


For image-to-text retrieval, run the commands first to transform text-to-image jsonls to image-to-text ones:
```bash
python src/eval/transform_ir_annotation_to_tr.py \
        --input ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl
```
After that，run the following commands
```bash
split=valid # validation / test splits
python src/eval/evaluation_tr.py \
        ${DATAPATH}/datasets/${dataset_name}/${split}_texts.tr.jsonl \
        ${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl \
        output.json
cat output.json
```

The printed results are shown below:
```json
{"success": true, "score": 85.67, "scoreJson": {"score": 85.67, "mean_recall": 85.67, "r1": 71.2, "r5": 90.5, "r10": 95.3}}
```
<br><br>

## Future Plans
+ Release pretrained Chinese-CLIP checkpoint with ViT-L-14 visual backbone (under training).
+ Provide online text-to-image retrieval demo based on Chinese-CLIP, with easy-to-use tutorials for deploying on users' own environments.
+ Obtain evaluation results on more cross-modal retrieval benchmarks.
+ Release technical report of Chinese-CLIP.
<br><br>


## Citation
If you find the project helpful, please star this project and cite the related articles. Thanks for your support!

```
@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}

@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```
