[**中文说明**](README.md) | [**English**](README_En.md)

<p align="center">
    <br>
    <img src="assets/Chinese_CLIP_logo_tp_path.svg" width="400" />
    <br>
<p>
<p align="center">
    <a href="https://opensource.org/licenses/MIT">
        <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
</p>

本项目为CLIP模型的**中文**版本，使用大规模中文数据进行训练（~2亿图文对），旨在帮助用户实现中文领域的跨模态检索、图像表示等。本项目代码基于<b>[open_clip project](https://github.com/mlfoundations/open_clip)</b>建设，并针对中文领域数据以及在中文数据上实现更好的效果做了优化。本项目提供了API、训练代码和测试代码，下文中将详细介绍细节。
<br><br>

## 新闻
* 2022.7.13 新增API功能，方便快速调用中文CLIP模型
* 2022.7.8 Chinese CLIP项目正式开源
<br><br>

## 实验结果
我们在MUGE Retrieval、Flickr30K-CN和COCO-CN上进行了zero-shot和finetune的实验，实验结果如下：

**MUGE Text-to-Image Retrieval**:
<table border="1" width="100%">
    <tr align="center">
        <th>Setup</th><th colspan="4">Zero-shot</th><th colspan="4">Finetune</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td>
    </tr>
	<tr align="center">
        <td width="120%">Wukong<sub>ViT-B</sub></td><td>33.4</td><td>59.3</td><td>69.7</td><td>54.1</td><td>39.2</td><td>66.9</td><td>77.4</td><td>61.2</td>
    </tr>
	<tr align="center">
        <td width="120%">R2D2<sub>ViT-B</sub></td><td>-</td><td>-</td><td>-</td><td>-</td><td>47.4</td><td>75.1</td><td>83.5</td><td>68.7</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B</sub></td><td>52.1</td><td>76.7</td><td>84.4</td><td>71.1</td><td>56.8</td><td>82.4</td><td>89.3</td><td>76.2</td>
    </tr>
</table>
<br>

**Flickr30K-CN Retrieval**:
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
        <td width="120%">Wukong<sub>ViT-B</sub></td><td>45.7</td><td>73.8</td><td>82.2</td></td><td>67.6</td><td>89.6</td><td>94.2</td><td>66.2</td><td>88.7</td><td>94.3</td></td><td>83.9</td><td>97.6</td><td>99.0</td>
    </tr>
	<tr align="center">
        <td width="120%">R2D2<sub>ViT-B</sub></td><td>-</td><td>-</td><td>-</td><td>78.3</td><td>94.6</td><td>97.0</td></td><td>-</td><td>-</td><td>-</td><td>92.6</td><td>99.1</td><td>99.8</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B</sub></td><td>57.0</td><td>82.8</td><td>89.6</td><td>72.9</td><td>92.8</td><td>96.2</td><td>71.0</td><td>90.6</td><td>95.3</td><td>88.6</td><td>98.0</td><td>99.3</td>
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
        <td width="120%">Wukong<sub>ViT-B</sub></td><td>49.2</td><td>79.4</td><td>87.9</td></td><td>67.0</td><td>91.4</td><td>96.7</td><td>48.3</td><td>77.8</td><td>88.8</td></td><td>65.8</td><td>90.3</td><td>96.6</td>
    </tr>
	<tr align="center">
        <td width="120%">R2D2<sub>ViT-B</sub></td><td>-</td><td>-</td><td>-</td><td>75.1</td><td>94.2</td><td>98.1</td></td><td>-</td><td>-</td><td>-</td><td>76.1</td><td>95.3</td><td>98.5</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B</sub></td><td>58.8</td><td>85.3</td><td>93.3</td><td>73.6</td><td>94.9</td><td>97.8</td><td>54.7</td><td>84.7</td><td>92.3</td><td>73.6</td><td>94.8</td><td>98.1</td>
    </tr>
</table>
<br><br>


## 安装要求
开始本项目前，需先检查是否满足下列环境配置要求:

* python >= 3.6.4
* pytorch >= 1.7.1 (with torchvision)
* CUDA Version >= 10.1

运行下列命令即可安装本项目所需的三方库。

```bash
pip install -r requirements.txt
```
<br><br>

## API快速上手
下面提供一段简单的代码示例说明如何使用中文CLIP的API。开始使用前，请先安装cn_clip：
```bash
# 安装最新的稳定版本
pip install cn_clip
# 或从源代码安装
cd Chinese-CLIP/
pip install -e .
```
安装成功后，即可通过如下方式轻松调用API：
```python
import torch 
from PIL import Image

import cn_clip.clip as clip
from cn_clip.clip import load_from_name
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(device)
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
```
如果你不满足于仅仅使用API，欢迎继续阅读本文档，了解如何使用我们的项目进行CLIP模型的训练和测试。
<br><br>


## 开始用起来！

### 代码组织

下载本项目后, 请创建新的文件夹 ```${DATAPATH}``` 以存放数据集、预训练ckpt、以及finetune产生的模型日志&ckpt。推荐工作区目录结构如下：

```
Chinese-CLIP/
├── run_scripts/
│   ├── muge_finetune_vit-b-16_rbt-base.sh
│   ├── flickr30k_finetune_vit-b-16_rbt-base.sh
│   └── ...           # 更多finetune或评测脚本...
└── cn_clip/
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
    └── .../          # 更多自定义数据集...
```

### 准备工作
这里我们提供预训练模型参数的下载方式，以及进行finetune前对数据进行的预处理过程

#### 预训练CKPT

目前我们提供ViT-B规模的预训练中文CLIP权重下载（[下载链接](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt)）。推荐将下载的ckpt文件存放于`${DATAPATH}/pretrained_weights/`目录下。

#### 数据集格式预处理

为了与Chinese-CLIP代码适配，同时保证数据处理和读取的效率，我们建议将训练&评测使用的图文数据集统一组织成如下的方式：

```
${DATAPATH}
└── datasets/
    └── ${dataset_name}/
        ├── train_imgs.tsv      # 图片id & 图片内容
        ├── train_texts.jsonl   # 文本id & 文本内容，连同匹配的图片id列表
        ├── valid_imgs.tsv
        ├── valid_texts.jsonl
        ├── test_imgs.tsv
        └── test_texts.jsonl
```
其中`${dataset_name}`代指数据集名称（如MUGE）

为保证文件处理效率，我们不是将图片以大量的小文件方式存放，而是将训练/验证/测试图片以base64形式分别存放在`${split}_imgs.tsv`文件中。文件每行表示一张图片，包含图片id（int型）与图片base64，以tab隔开，格式如下：
```
1000002	/9j/4AAQSkZJ...YQj7314oA//2Q==
```

将图片原始文件转换为base64的方式非常简单，请执行以下python代码：
```python
from PIL import Image
from io import BytesIO
import base64

img = Image.open(file_name) # 访问图片路径
img_buffer = BytesIO()
img.save(img_buffer, format=img.format)
byte_data = img_buffer.getvalue()
base64_str = base64.b64encode(byte_data) # bytes
base64_str = base64_str.decode("utf-8") # str
```

文本信息及图文对匹配关系则保存在`${split}_texts.jsonl`文件。文件每行是一行json，格式如下：
```
{"text_id": 8428, "text": "高级感托特包斜挎", "image_ids": [1076345, 517602]}
```

最后，我们还需要将tsv和jsonl文件一起序列化，转换为内存索引的LMDB数据库文件，方便训练时的随机读取
```
python cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir ${DATAPATH}/datasets/${dataset_name}
    --splits train,valid,test
```
例如对于MUGE数据集，则`${dataset_name}`设为MUGE，`--splits`指定需要转换的数据集划分，以逗号不加空格分隔。转换后，数据集文件夹下会对应增加以下LMDB序列化文件
```
${DATAPATH}
└── datasets/
    └── ${dataset_name}/
        └── lmdb/
            ├── train
            │   ├── imgs
            │   └── pairs
            ├── valid
            └── test
```

### 模型finetune

在此我们介绍训练的步骤，方便其他用户了解模型细节，使用我们提供的中文CLIP预训练模型进行finetune。基于MUGE和Flickr30K-CN两个下游检索数据集，我们提供了训练样例脚本`run_scripts/muge_finetune_vit-b-16_rbt-base.sh`和`run_scripts/flickr30k_finetune_vit-b-16_rbt-base.sh`。运行脚本同时支持单机和多机分布式训练，请在运行前，先根据脚本开头的指引注释，填写好分布式相关配置，之后运行如下命令即可开始训练（多机训练请在各机器上都运行命令）。训练产生的log和模型ckpt文件，会自动保存在用户指定的目录下：

```bash
cd Chinese-CLIP/
bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}
```

相关的训练配置项包括:

+ 分布式
  + `WORKER_CNT`: 训练的机器个数
  + `GPUS_PER_NODE`: 每个机器上的GPU个数
+ 训练/验证数据
  + `train-data`: 训练数据LMDB目录，准备LMDB数据文件的预处理流程见上。
  + `val-data`: 验证数据LMDB目录。
  + `num-workers`: 训练数据处理（DataLoader）的进程数，默认为4
+ 训练超参数
  + `vision-model`: 指定视觉backbone, 从 `["ViT-B-32", "ViT-B-16", "ViT-L-14"]`选择。
  + `text-model`: 指定文本backbone, 从 `["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese"]`选择。
  + `context-length`: 文本输入序列长度。
  + `warmup`: warmup步数。
  + `batch-size`: 训练时单卡batch-size。（请保证`训练样本总数 > batch-size * GPU数`，至少满足1个训练batch）
  + `lr`: 学习率。
  + `wd`: weight decay。
  + `max-steps`: 训练步数，也可通过`max-epochs`指定训练轮数。
  + `freeze-vision`: 是否freeze视觉backbone。
  + `use-augment`: 是否使用[AutoAugment](https://arxiv.org/abs/1805.09501)对图片进行数据增强
  + `valid-batch-size`: 验证时单机batch-size。（请保证`验证集样本总数 > batch-size * GPU数`，至少满足1个验证batch）
  + `valid-step-interval`和`valid-epoch-interval`: 验证step/epoch频率，指定为-1时则在训练中不进行验证
+ 输出选项
  + `name`: 指定输出路径。超参日志, 训练日志以及产出ckpt均会存放至 `${DATAPATH}/experiments/${name}/`。
  + `save-step-frequency`及`save-epoch-frequency`: 存ckpt的步数或轮数间隔。
  + `report-training-batch-acc`: 日志是否报告训练图到文&文到图batch准确率。
+ 权重读取相关选项
  + `resume`: 权重读取的路径。示例脚本中指定为预训练ckpt路径，也可以指定为用户自己finetune的ckpt路径做继续训练。
  + `reset-data-offset`: 是否从此前的数据断点续跑。如batch size或GPU卡数超参改变，建议打开此选项。
  + `reset-optimizer`: 是否使用optimizer state。

训练完毕，log 会自动存在`${DATAPATH}/experiments/${name}/out_${timestamp}.log`，训练log格式如下所示:
```
2022-06-16,10:58:27 | INFO | Rank 0 | Global Steps: 1/735 | Train Epoch: 1 [1024/250880 (0%)] | Loss: 2.171807 | Image2Text Acc: 49.41 | Text2Image Acc: 52.54 | Data Time: 5.167s | Batch Time: 15.647s | LR: 0.000000 | logit_scale: 4.605 | Global Batch Size: 1024
```
验证log格式如下所示:
```
2022-06-16,11:06:00 | INFO | Rank 0 | Validation Result (epoch 1 @ 150 steps) | Valid Loss: 0.503617 | Image2Text Acc: 84.76 | Text2Image Acc: 84.37 | logit_scale: 4.605 | Valid Batch Size: 128
```

**注意**: 对比学习的训练收敛和稳定性和总batch size相关。如您使用更小的batch size（相比默认配置128 per-GPU \* 8 GPU），建议使用更小的学习率。我们推荐使用更多的GPU和更大的batch size以取得更好的效果。

### 预测及评估

我们提供特征提取、以及图文检索任务评估的流程，具体如下：

#### 图文特征提取

目前本代码支持使用GPU单卡进行图文特征提取，请参考使用以下命令
```bash
cd Chinese-CLIP/
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

split=valid # 指定计算valid或test集特征
resume=${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt

python -u cn_clip/eval/extract_features.py \
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

产出图文特征默认将保存于`${DATAPATH}/datasets/${dataset_name}`目录下，图片特征保存于`${split}_imgs.img_feat.jsonl`文件，每行以json存储一张图片的特征，格式如下：
```
{"image_id": 1000002, "feature": [0.0198, ..., -0.017, 0.0248]}
```
文本特征则保存于`${split}_texts.txt_feat.jsonl`，格式如下：
```
{"text_id": 248816, "feature": [0.1314, ..., 0.0018, -0.0002]}
```

#### KNN检索

对于小规模的学术检索数据集，我们提供一个简单的KNN检索实现，便于计算文到图、图到文检索的top-k召回结果

对于文到图检索（文本召回相关图片），请运行以下命令：
```bash
cd Chinese-CLIP/
split=valid # 指定计算valid或test集特征
python -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl"
```
产出的结果保存在指定的jsonl文件中，每行表示一个文本召回的top-k图片id，格式如下：
```json
{"text_id": 153915, "image_ids": [5791244, 1009692167, 7454547004, 3564007203, 38130571, 2525270674, 2195419145, 2503091968, 4966265765, 3690431163]}
```

对于图到文检索（图片召回相关文本），类似地，请运行以下命令：
```bash
split=valid # 指定计算valid或test集特征
python -u cn_clip/eval/make_topk_predictions_tr.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl"
```
产出结果每行表示一个图片召回的top-k文本id，格式如下：
```json
{"image_id": 977856234, "text_ids": [156914, 157914, 158914, 155914, 156179, 158907, 157179, 154179, 154914, 154723]}
```

#### Recall计算

我们提供了评测脚本计算检索任务的Recall@1/5/10，同时给出mean recall（Recall@1/5/10的平均数）。运行如下命令即可获取分数:

对于文到图检索，请运行命令：
```bash
split=valid # 指定计算valid或test集特征
python cn_clip/eval/evaluation.py \
        ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl \
        ${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl \
        output.json
cat output.json
```

对于图到文检索，请先运行下面的命令，将图文对标注的jsonl文件由文到图的格式转为图到文：
```bash
python cn_clip/eval/transform_ir_annotation_to_tr.py \
        --input ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl
```
完成后，请运行命令：
```bash
split=valid # 指定计算valid或test集特征
python cn_clip/eval/evaluation_tr.py \
        ${DATAPATH}/datasets/${dataset_name}/${split}_texts.tr.jsonl \
        ${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl \
        output.json
cat output.json
```
打印出的结果格式将如下：
```json
{"success": true, "score": 85.67, "scoreJson": {"score": 85.67, "mean_recall": 85.67, "r1": 71.2, "r5": 90.5, "r10": 95.3}}
```
<br><br>

## 后续计划
+ 开源ViT-L-14规模Chinese-CLIP模型（训练中）
+ 提供基于Chinese-CLIP的图文检索demo，以及用户在自己的环境下部署demo的流程
+ 在更多图文检索下游任务验证结果
+ 开源Chinese-CLIP技术报告
<br><br>

## 引用
如果觉得本项目好用，希望能给我们提个star并分享给身边的用户，欢迎给相关工作citation，感谢支持！

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
