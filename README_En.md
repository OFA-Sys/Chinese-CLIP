[**中文说明**](README.md) | [**English**](README_En.md)

<p align="center">
    <br>
    <img src="assets/Chinese_CLIP_logo_tp_path.svg" width="400" />
    <br>
<p>
<br>

<p align="center">
        <a href="https://www.modelscope.cn/models?name=clip&tasks=multi-modal-embedding">ModelScope</a>&nbsp ｜ &nbsp<a href="https://www.modelscope.cn/studios/damo/chinese_clip_applications/summary">Demo</a>&nbsp ｜ &nbsp<a href="https://arxiv.org/abs/2211.01335">Paper </a>&nbsp ｜ &nbspBlog
</p>
<br><br>

This is the Chinese version of CLIP. We use a large-scale Chinese image-text pair dataset (~200M) to train the model, and we hope that it can help users to conveniently achieve [image representation generation](#api-use-case), [cross-modal retrieval](#cross-modal-retrieval) and [zero-shot image classification](#zero-shot-image-classification) for Chinese data. This repo is based on <b>[open_clip project](https://github.com/mlfoundations/open_clip)</b>. We have made some optimization for better performance on Chinese data, and we provide the details in the following. 
<br><br>

# News
* 2023.9.8 Chinese-CLIP has supported [knowledge distillation fine-tuning](distillation_En.md) based on [ModelScope](https://github.com/modelscope/modelscope). (Thanks [@wuziheng](https://github.com/wuziheng) and [@Jaskr616](https://github.com/Jaskr616) from Aliyun PAI Team for [the PR](https://github.com/OFA-Sys/Chinese-CLIP/pull/195)❤️)
* 2023.5.9 Chinese-CLIP has been adapted to Pytorch2.0.
* 2023.3.20 Support [gradient accumulation](#gradient-accumulation) in contrastive learning to simulate the training effect of a larger batch size.
* 2023.2.16 Support [FlashAttention](https://github.com/HazyResearch/flash-attention) to improve training speed and reduce memory usage. See [flash_attention_En.md](flash_attention_En.md) for more information.
* 2023.1.15 Support the conversion of Pytorch models into [ONNX](https://onnx.ai/) or [TensorRT](https://developer.nvidia.com/tensorrt) formats (and provide pretrained TensorRT models) to improve inference speed and meet deployment requirements. See [deployment_En.md](deployment_En.md) for more information.
* 2022.12.12 Implement [FLIP](https://arxiv.org/abs/2212.00794) strategy, which can be [activated](#FLIP) during finetuning (Thanks [@zwkkk](https://github.com/zwkkk) for [the PR](https://github.com/OFA-Sys/Chinese-CLIP/pull/26) ❤️）
* 2022.12.3 The datasets of the Chinese version of the [ELEVATER](https://eval.ai/web/challenges/challenge-page/1832) benchmark are publicly available. See [Notes for datasets](zeroshot_dataset_en.md) for more information. 
* 2022.12.1 Chinese-CLIP model & representation generation API are officially merged into Huggingface transformers🤗 codebase.
* 2022.11.22 Release [zero-shot image classification](#zero-shot-image-classification) code. Support [ELEVATER](https://eval.ai/web/challenges/challenge-page/1832) zero-shot classification benchmark.
* 2022.11.3 Release RN50, ViT-H-14 models. Release [technical report](https://arxiv.org/pdf/2211.01335.pdf).
* 2022.9.22 Release ViT-L-14, ViT-L-14-336 models.
* 2022.7.13 Release [fast image & text representation generation API](#api-use-case), which facitilates usage of our CLIP models quickly.
* 2022.7.8 Release the project Chinese-CLIP! Release [image-text retrieval](#cross-modal-retrieval) code.
<br><br>

# Models and Results
<span id="model_card"></span>
## Model Card
Currently, we release 5 different sizes of Chinese-CLIP models. Detailed information and download link of each Chinese-CLIP model are provided below:

<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>Ckpt</th><th>#Params (All)</th><th>Backbone (I)</th><th>#Params (I)</th><th>Backbone (T)</th><th>#Params (T)</th><th>Resolution</th>
    </tr>
    <tr align="center">
        <td>CN-CLIP<sub>RN50</sub></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_rn50.pt">Download</a></td><td>77M</td><td>ResNet50</td><td>38M</td><td>RBT3</td><td>39M</td><td>224</td>
    </tr>
    <tr align="center">
        <td>CN-CLIP<sub>ViT-B/16</sub></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt">Download</a></td><td>188M</td><td>ViT-B/16</td><td>86M</td><td>RoBERTa-wwm-Base</td><td>102M</td><td>224</td>
    </tr>
    <tr align="center">
        <td>CN-CLIP<sub>ViT-L/14</sub></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-l-14.pt">Download</a></td><td>406M</td><td>ViT-L/14</td><td>304M</td><td>RoBERTa-wwm-Base</td><td>102M</td><td>224</td>
    </tr>
    <tr align="center">
        <td>CN-CLIP<sub>ViT-L/14@336px</sub></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-l-14-336.pt">Download</a></td><td>407M</td><td>ViT-L/14</td><td>304M</td><td>RoBERTa-wwm-Base</td><td>102M</td><td>336</td>
    </tr>
    <tr align="center">
        <td>CN-CLIP<sub>ViT-H/14</sub></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-h-14.pt">Download</a></td><td>958M</td><td>ViT-H/14</td><td>632M</td><td>RoBERTa-wwm-Large</td><td>326M</td><td>224</td>
    </tr>
</table>
<br></br>

## Results
We conducted zero-shot inference and finetuning experiments on [MUGE Retrieval](https://tianchi.aliyun.com/muge), [Flickr30K-CN](https://github.com/li-xirong/cross-lingual-cap) and [COCO-CN](https://github.com/li-xirong/coco-cn) for the evaluation of cross-modal retrieval, and conducted experiments on 10 image classification datasets of the [ELEVATER](https://eval.ai/web/challenges/challenge-page/1832) benchmark for the evaluation of zero-shot image classification. Results are shown below. Due to space limitation, here we only list the performance of the best performing Chinese-CLIP and baseline models. For detailed performance of each Chinese-CLIP model size, please refer to [Results.md](Results.md).

**MUGE Text-to-Image Retrieval (Official Validation Set)**:
<table border="1" width="100%">
    <tr align="center">
        <th>Setup</th><th colspan="4">Zero-shot</th><th colspan="4">Finetune</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td>
    </tr>
	<tr align="center">
        <td width="120%">Wukong</td><td>42.7</td><td>69.0</td><td>78.0</td><td>63.2</td><td>52.7</td><td>77.9</td><td>85.6</td><td>72.1</td>
    </tr>
	<tr align="center">
        <td width="120%">R2D2</td><td>49.5</td><td>75.7</td><td>83.2</td><td>69.5</td><td>60.1</td><td>82.9</td><td>89.4</td><td>77.5</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP</td><td>63.0</td><td>84.1</td><td>89.2</td><td>78.8</td><td>68.9</td><td>88.7</td><td>93.1</td><td>83.6</td>
    </tr>
</table>
<br>

**Flickr30K-CN Retrieval (Official Test Set)**:
<table border="1" width="120%">
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
        <td width="120%">Wukong</td><td>51.7</td><td>78.9</td><td>86.3</td><td>77.4</td><td>94.5</td><td>97.0</td><td>76.1</td><td>94.8</td><td>97.5</td><td>92.7</td><td>99.1</td><td>99.6</td>
    </tr>
	<tr align="center">
        <td width="120%">Taiyi</td><td>60.8</td><td>85.0</td><td>91.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>	
	<tr align="center">
        <td width="120%">R2D2</td><td>60.9</td><td>86.8</td><td>92.7</td><td>84.4</td><td>96.7</td><td>98.4</td><td>77.6</td><td>96.7</td><td>98.9</td><td>95.6</td><td>99.8</td><td>100.0</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP</td><td>71.2</td><td>91.4</td><td>95.5</td><td>83.8</td><td>96.9</td><td>98.6</td><td>81.6</td><td>97.5</td><td>98.8</td><td>95.3</td><td>99.7</td><td>100.0</td>
    </tr>
</table>
<br>

**COCO-CN Retrieval (Official Test Set)**:
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
        <td width="120%">Wukong</td><td>53.4</td><td>80.2</td><td>90.1</td><td>74.0</td><td>94.4</td><td>98.1</td><td>55.2</td><td>81.0</td><td>90.6</td><td>73.3</td><td>94.0</td><td>98.0</td>
    </tr>
	<tr align="center">
        <td width="150%">Taiyi</td><td>60.0</td><td>84.0</td><td>93.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>	
	<tr align="center">
        <td width="120%">R2D2</td><td>56.4</td><td>85.0</td><td>93.1</td><td>79.1</td><td>96.5</td><td>98.9</td><td>63.3</td><td>89.3</td><td>95.7</td><td>79.3</td><td>97.1</td><td>98.7</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP</td><td>69.2</td><td>89.9</td><td>96.1</td><td>81.5</td><td>96.9</td><td>99.1</td><td>63.0</td><td>86.6</td><td>92.9</td><td>83.5</td><td>97.3</td><td>99.2</td>
    </tr>
</table>
<br>

**Zero-shot Image Classification**:
<table border="1" width="100%">
	<tr align="center">
        <th>Task</th><th>CIFAR10</th><th>CIFAR100</th><th>DTD</th><th>EuroSAT</th><th>FER</th><th>FGVC</th><th>KITTI</th><th>MNIST</th><th>PC</th><th>VOC</th>
    </tr>
	<tr align="center">
        <td width="150%">GIT</td><td>88.5</td><td>61.1</td><td>42.9</td><td>43.4</td><td>41.4</td><td>6.7</td><td>22.1</td><td>68.9</td><td>50.0</td><td>80.2</td>
    </tr>
    	<tr align="center">
        <td width="150%">ALIGN</td><td>94.9</td><td>76.8</td><td>66.1</td><td>52.1</td><td>50.8</td><td>25.0</td><td>41.2</td><td>74.0</td><td>55.2</td><td>83.0</td>
    </tr>
	<tr align="center">
        <td width="150%">CLIP</td><td>94.9</td><td>77.0</td><td>56.0</td><td>63.0</td><td>48.3</td><td>33.3</td><td>11.5</td><td>79.0</td><td>62.3</td><td>84.0</td>
    </tr>
    	<tr align="center">
        <td width="150%">Wukong</td><td>95.4</td><td>77.1</td><td>40.9</td><td>50.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    	<tr align="center">
        <td width="150%">CN-CLIP</td><td>96.0</td><td>79.7</td><td>51.2</td><td>52.0</td><td>55.1</td><td>26.2</td><td>49.9</td><td>79.4</td><td>63.5</td><td>84.9</td>
    </tr>
</table>
<br><br>

# Getting Started
## Installation Requirements
To start with this project, make sure that your environment meets the requirements below:

* python >= 3.6.4
* pytorch >= 1.8.0 (with torchvision >= 0.9.0)
* CUDA Version >= 10.2

Run the following command to install required packages.

```bash
pip install -r requirements.txt
```

## API Use Case
We provide a simple code snippet to show how to use the API for Chinese-CLIP. For starters, please install cn_clip:
```bash
# to install the latest stable release
pip install cn_clip

# or install from source code
cd Chinese-CLIP
pip install -e .
```
After installation, use Chinese CLIP to compute the image ([example](examples/pokemon.jpeg)) & text embeddings and similarities as shown below:
```python
import torch 
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(device)
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    # Normalize the features. Please use the normalized features for downstream tasks.
    image_features /= image_features.norm(dim=-1, keepdim=True) 
    text_features /= text_features.norm(dim=-1, keepdim=True)      

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
```

However, if you are not satisfied with only using the API, move on for more details about training and inference. 
<br><br>


# Tutorial

Currently, we provide the tutorial of [cross-modal retrieval](#cross-modal-retrieval) and [zero-shot image classification](#zero-shot-image-classification) below.

## Cross-Modal Retrieval

### Code Organization

After cloning this project, please create a new directory ```${DATAPATH}``` for datasets, checkpoints and logs. A recommended workspace structure is demonstrated below：

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
├── deploy/	      # store ONNX & TensorRT deployment models
└── datasets/
    ├── MUGE/
    ├── Flickr30k-CN/
    └── .../          # more datasets...
```

### Preparation
We provide links for the downloading of pretrained checkpoints, as well as the data preprocessing procedures for finetuning. 

#### Pretrained Checkpoints

Please refer to [model card section](#model_card) above and download the model checkpoint. We recommend putting the checkpoint in `${DATAPATH}/pretrained_weights/`. 

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
For the test set where only the texts are given and the image-text pairing relations are unknown, just leave the `image_ids` field as an empty list, `"image_ids": []`.

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

For easier use, we have provided preprocessed MUGE ([download link](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/MUGE.zip)) and Flickr30K-CN ([download link](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/Flickr30k-CN.zip)) datasets in zip format. To use them, just download and unzip it under `${DATAPATH}/datasets/`. If you need [COCO-CN](https://github.com/li-xirong/coco-cn) dataset, please contact us by email when you have finished applying for permission from the original author.

### Finetuning

We introduce the procedures of training for users to learn about the details of the model. We finetune with the pretrained Chinese CLIP. For MUGE and Flickr30K-CN, we provide scripts `run_scripts/muge_finetune_vit-b-16_rbt-base.sh` and `run_scripts/flickr30k_finetune_vit-b-16_rbt-base.sh`. <b>The scripts support single-worker and distributed training. Before running, follow the instructions at the beggining of the scripts and fill in your configuration for distributed training. Then run the scripts to start your training. If the GPU memory is insufficient, you can consider to [activate the gradient checkpointing strategy](#checkpointing) in the configuration.</b> Logs and checkpoints will be saved at your specified paths. 

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
  + `val-data`: directory of validation data. If set to None, validation during finetuning will be disabled.
  + `num-workers`: the number of workers for training set dataloader, default to 4.
  + `valid-num-workers`: the number of workers for validation set dataloader, default to 1.
+ Training hyper-params
  + `vision-model`: specified visual backbones. Select from `["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"]`.
  + `text-model`: specified language backbones. Select from `["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"]`.
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
  + `grad-checkpointing`: <span id="checkpointing"></span>use [gradient checkpointing]((https://pytorch.org/docs/stable/checkpoint.html)) which does not keep the activations during forward computation, this strategy trades more computation and iteration time for less GPU memory cost. (`store_true` argument, just add `--grad-checkpointing` in the script to activate it, requires Pytorch>1.8.0)
  + `mask-ratio`: <span id="FLIP"></span>use [FLIP](https://arxiv.org/abs/2212.00794) strategy which randomly masks a ratio of image patches to save GPU memory and speed up training. Default to 0.0, which disables the strategy.
  + `use-flash-attention`: whether to use [FlashAttention](https://arxiv.org/abs/2205.14135), which can significantly speed up the finetune process and reduce the memory usage. (`store_true` argument, after configuring the environment, just add `--use-flash-attention` in the script to activate it, please see [flash_attention_En.md](flash_attention_En.md) for more information)
  + `accum-freq`: <span id="gradient-accumulation"></span>Gradient accumulation frequency, default is 1. Specify an integer greater than 1 to enable gradient accumulation to simulate a larger batch size. if the batch size for a worker is `m`, the total batch size is `accum_freq * m * GPUs`.
  + `gather-with-grad`: Whether to enable full distributed gradient for feature gather, off by default.
+ Ouputs
  + `name`: specified output path. Hyperparameter logs, training logs, and checkpoints will be saved at `${DATAPATH}/experiments/${name}/`.
  + `save-step-frequency` and `save-epoch-frequency`: the intervals for saving checkpoints.
  + `report-training-batch-acc`: whether to report the in-batch image-to-text and text-to-image retrieval accuracy. 
+ Checkpoints
  + `resume`: the checkpoint path for weights to restore. In the provided example script, the path refers to the pretrained checkpoint path. Users can change to your own checkpoint path.
  + `reset-data-offset`: whether to restore training at the data breakpoint.
  + `reset-optimizer`: whether to restore the optimizer state.

After training, the log will be saved at `${DATAPATH}/experiments/${name}/out_${timestamp}.log`. Example of log is shown below:
```
2022-12-11,20:40:34 | INFO | Rank 0 | Global Steps: 1/735 | Train Epoch: 1 [1024/250880 (0%)] | Loss: 2.371020 | Image2Text Acc: 49.90 | Text2Image Acc: 48.73 | Data Time: 1.039s | Batch Time: 3.625s | LR: 0.000000 | logit_scale: 4.605 | Global Batch Size: 1024
```
The example of validation log is shown below:
```
2022-12-11,20:42:47 | INFO | Rank 0 | Validation Result (epoch 1 @ 150 steps) | Valid Loss: 0.502810 | Image2Text Acc: 84.95 | Text2Image Acc: 84.26 | logit_scale: 4.605 | Valid Batch Size: 128
```

**Attention**: The convergence and stability of contrastive learning is highly relevant to the total batch size. If you use a smaller batch size, (in comparison with the default 128 per-GPU \* 8 GPU), we advise you to use a smaller learning rat. We recommend using more GPUs and larger batch size for better performance. 

### Inference and Evaluation

We provide procedures for representation generation and cross-modal retrieval, as demonstrated below:

#### Image/Text Representation Generation

By now the code supports representation generation with a single worker, please use the following commands. Besides, we provide support for deploying ONNX and TensorRT models to accelerate feature inference, see [deployment_En.md](deployment_En.md) for details.
```bash
cd Chinese-CLIP/
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/src

split=valid # validation / test set
resume=${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt

python -u src/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=52 \
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

For small-scale retrieval datasets, we provide a simple implementation of KNN retrieval, to facilitate the retrieval of top-k results in cross-modal retrieval. (tips: If you want to build a [retrieval demo](https://www.modelscope.cn/studios/damo/chinese_clip_applications/summary) in your project like us, we suggest first to use Chinese-CLIP to compute image and text embeddings, and then employ an opensource servering framework [clip-retrieval](https://github.com/rom1504/clip-retrieval) to deploy the front-end and back-end servering.)

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
After that, run the following commands
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

For better understanding of cross-modal retrieval by Chinese-CLIP, we also provide a runnable jupyter notebook ([download link](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/others/Chinese-CLIP-on-MUGE-Retrieval.ipynb)), which works with the MUGE retrieval dataset (corresponding leaderboard is hosted on [Tianchi](https://tianchi.aliyun.com/competition/entrance/532031/introduction?lang=en-us)) and includes the finetuning and inference process mentioned above. Welcome to try!

<br>

## Zero-shot Image Classification
This section introduces the use of Chinese-CLIP for zero-shot image classification. We use the experiment on a dataset of the benchmark ELEVATER as an example. ELEVATER is a benchmark consist of several widely used classification datasets and evaluates the zero-shot performance on these datasets, including CIFAR-10, CIFAR-100, MNIST, etc. In our experiments, we have perpared Chinese prompts and label names with the original images for each ELEVATER dataset (refer to [Notes for datasets](zeroshot_dataset_en.md) for download) to evaluate Chinese-CLIP. For more information about ELEVATER, please click [this link](https://eval.ai/web/challenges/challenge-page/1832/overview). Users can also follow the procedure below to prepare and evaluate their own classification datasets.
<br><br>

### Preparation
We need to prepare only the test set and the pretrained Chinese-CLIP checkpoint. It's recommended to prepare these directories under a user defined `${DATAPATH}` and organize them as follows:
```
${DATAPATH}
├── pretrained_weights/
└── datasets/
    └── ${dataset_name}/
        ├── label_cn.txt
        └── test/
	    ├── 000/ # label id，fill 0 by the left to 3 digits so that the labels can be alphabetically ordered
	    │   ├── image_0003.jpg # image sample, no specific requirements for the naming
	    │   ├── image_0005.jpg
	    │   └── ...
	    ├── 001/
	    │   ├── image_0001.jpg
	    │   ├── image_0002.jpg
	    │   └── ...
	    └── 002/
	        ├── image_0003.jpg
	        ├── image_0005.jpg
	        └── ...
	    ...
	
```
Make sure the data are categorized by their label id, and make sure the ids are alphabetically orderd (for numbers larger than 10, use`label.zfill(3)` to fill 0 by the left to 3 digits, like 001，002, etc). `label_cn.txt` refers to the file of label names. Each line has a label name, as demonstrated below:
```
accordion
airplane
anchor
...
```
The label id is `[line number]-1`. For example, the label id for the first line is 0, and the one for the second line is 1. If the number of labels is larger than 10, all labels are filled with 0 by the left to 3-digit numbers. For example, if the number of labels is 100, the ids are `000-099`. Users should create a directory for each label, and put the corresponding samples into the directories. We provide the processed dataset CIFAR-100 as an example, and please click [this link](http://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/cifar-100.zip) to download the prepared dataset. To evaluate other datasets of ELEVATER, please refer to [Notes for datasets](zeroshot_dataset_en.md) for download.
<br><br>

### Prediction and Evaluation
We provide a script for prediction and evaluation. Please check `run_scripts/zeroshot_eval.sh` for more details. An example command is shown below:
```bash
bash run_scripts/zeroshot_eval.sh 0 \
   ${DATAPATH} ${dataset_name} \
   ${vision_model} ${text_model} \
   ${ckpt_path} ${index_file}
```
where the arguments stand for:
+ the first argument `0` refers to the GPU ID
+ `DATAPATH` refers to the root directory storing the checkpoint and dataset, as mentioned in Preparation part above
+ `dataset_name` refers to the directory name of the dataset, e.g. cifar-100, as mentioned in Preparation part above
+ `vision_model` refers to the type of vision encoder, including `["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "RN50", "ViT-H-14"]`
+ `text_model` refers to the type of text encoder, including `["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"]`
+ `ckpt_path` refers to the complete path of the pretrained Chinese-CLIP checkpoint
+ `index_file` is optional and only needed when you would like to submit to ELEVATER official website. Please refer to [Notes for datasets](zeroshot_dataset_en.md) for more details

For example, to evaluate ViT-B/16 on CIFAR-100, please run (the `${DATAPATH}` should be replaced with your real path):
```bash
bash run_scripts/zeroshot_eval.sh 0 \
    ${DATAPATH} cifar-100 \
    ViT-B-16 RoBERTa-wwm-ext-base-chinese \
    ${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt
```

Top-1 accuracy will be printed. 
```
Result:
zeroshot-top1: 0.6444
```
On CIFAR-100, the ViT-B/16 model of Chinese-CLIP will achieve the accuracy of 64.4%. For the zero-shot evaluation results of other model scales and other datasets, please refer to [Results.md](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/Results.md#zeroshot_results).

Also, a json file will be saved, which serves the submission of ELEVATER. An example of the json file is shown below：
```json
{"model_name": "CN-CLIP-ViT-B-16", "dataset_name": "cifar-100", "num_trainable_params": 0, "num_params": 188262913, "num_visual_params": 86192640, "num_backbone_params": 188262913, "n_shot": 0, "rnd_seeds": [123], "predictions": "prediction probability tensor [size: (1, 10000, 100)]"}
```
It includes meta data like the name of model `model_name`, the dataset name `dataset_name`, the number of parameters`num_params`, the number of parameters of vision encoder `num_visual_params`, and also the outputs of the model, namely the predicted probability tensor, whose size is `[1, num_samples, num_labels]`. 

### Zero-Shot Classification Online Demo
Based on the representation generation API which we have integrated into Huggingface transformers, we are able to provide online demos of zero-shot classification task on Huggingface Model Hub🤗 for each scale of Chinese-CLIP model. The links are given below:
- [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
- [OFA-Sys/chinese-clip-vit-large-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14)
- [OFA-Sys/chinese-clip-vit-large-patch14-336px](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14-336px)
- [OFA-Sys/chinese-clip-vit-huge-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14)
- **（Update on 12.10🔥）**[**New version of demo deployed on Huggingface Spaces**](https://huggingface.co/spaces/OFA-Sys/chinese-clip-zero-shot-image-classification): the 4 model scales above are all gathered into the same demo page, supporting customed prompt template by user. **Welcome to try!**
<br><br><br>


# Citation
If you find the project helpful, please star this project and cite the related articles. Thanks for your support!

```
@article{chinese-clip,
  title={Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese},
  author={Yang, An and Pan, Junshu and Lin, Junyang and Men, Rui and Zhang, Yichang and Zhou, Jingren and Zhou, Chang},
  journal={arXiv preprint arXiv:2211.01335},
  year={2022}
}
```
