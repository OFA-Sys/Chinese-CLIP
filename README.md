[**中文说明**](README.md) | [**English**](README_En.md)

<p align="center">
    <br>
    <img src="assets/Chinese_CLIP_logo_tp_path.svg" width="400" />
    <br>
<p>
<br>

<p align="center">
        <a href="https://www.modelscope.cn/models?name=clip&tasks=multi-modal-embedding">ModelScope</a>&nbsp ｜ &nbsp<a href="https://www.modelscope.cn/studios/damo/chinese_clip_applications/summary">Demo</a>&nbsp ｜ &nbsp<a href="https://arxiv.org/abs/2211.01335">Paper</a>&nbsp ｜ &nbspBlog
</p>
<br><br>

本项目为CLIP模型的**中文**版本，使用大规模中文数据进行训练（~2亿图文对），旨在帮助用户快速实现中文领域的[图文特征&相似度计算](#API快速上手)、[跨模态检索](#跨模态检索)、[零样本图片分类](#零样本图像分类)等任务。本项目代码基于<b>[open_clip project](https://github.com/mlfoundations/open_clip)</b>建设，并针对中文领域数据以及在中文数据上实现更好的效果做了优化。本项目提供了API、训练代码和测试代码，下文中将详细介绍细节。
<br><br>

# 新闻
* 2022.12.3 公开[ELEVATER](https://eval.ai/web/challenges/challenge-page/1832)图像分类数据集中文版本，详见[数据文档](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/zeroshot_dataset.md)
* 2022.12.1 Chinese-CLIP模型代码&特征提取API，同步合入Huggingface transformers🤗代码库
* 2022.11.22 新增[零样本图像分类](#零样本图像分类)代码，可支持[ELEVATER benchmark](https://eval.ai/web/challenges/challenge-page/1832)零样本分类评测任务
* 2022.11.3 新增RN50，ViT-H-14模型，公开[技术报告](https://arxiv.org/pdf/2211.01335.pdf)
* 2022.9.22 新增ViT-L-14，ViT-L-14-336模型
* 2022.7.13 新增[图文特征提取快速API](#API快速上手)，几行代码快速调用中文CLIP模型，计算图文特征&相似度
* 2022.7.8 Chinese-CLIP项目正式开源，开源[图文检索](#跨模态检索)代码
<br><br>

# 模型及实验
<span id="model_card"></span>
## 模型规模 & 下载链接
Chinese-CLIP目前开源5个不同规模，其模型信息和下载方式见下表：

<table border="1" width="100%">
    <tr align="center">
        <th>模型规模</th><th>下载链接</th><th>参数量</th><th>视觉侧骨架</th><th>视觉侧参数量</th><th>文本侧骨架</th><th>文本侧参数量</th><th>分辨率</th>
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

## 实验结果
针对图文检索任务，我们在[MUGE Retrieval](https://tianchi.aliyun.com/muge)、[Flickr30K-CN](https://github.com/li-xirong/cross-lingual-cap)和[COCO-CN](https://github.com/li-xirong/coco-cn)上进行了zero-shot和finetune的实验。针对图像零样本分类，我们在[ELEVATER](https://eval.ai/web/challenges/challenge-page/1832)的10个数据集上进行了实验。实验结果如下表所示。篇幅所限，我们这里给出baseline模型和Chinese-CLIP的最大规模模型结果，关于Chinese-CLIP各规模的详细结果指标，请详见[Results.md](Results.md)。

**MUGE Text-to-Image Retrieval**:
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

**Flickr30K-CN Retrieval**:
<table border="1" width="150%">
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
        <td width="120%">R2D2</td><td>60.9</td><td>86.8</td><td>92.7</td><td>84.4</td><td>96.7</td><td>98.4</td><td>77.6</td><td>96.7</td><td>98.9</td><td>95.6</td><td>99.8</td><td>100.0</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP</td><td>71.2</td><td>91.4</td><td>95.5</td><td>83.8</td><td>96.9</td><td>98.6</td><td>81.6</td><td>97.5</td><td>98.8</td><td>95.3</td><td>99.7</td><td>100.0</td>
    </tr>
</table>
<br>

**COCO-CN Retrieval**:
<table border="1" width="150%">
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
        <td width="150%">Wukong</td><td>53.4</td><td>80.2</td><td>90.1</td><td>74.0</td><td>94.4</td><td>98.1</td><td>55.2</td><td>81.0</td><td>90.6</td><td>73.3</td><td>94.0</td><td>98.0</td>
    </tr>
	<tr align="center">
        <td width="150%">R2D2</td><td>56.4</td><td>85.0</td><td>93.1</td><td>79.1</td><td>96.5</td><td>98.9</td><td>63.3</td><td>89.3</td><td>95.7</td><td>79.3</td><td>97.1</td><td>98.7</td>
    </tr>
	<tr align="center">
        <td width="150%">CN-CLIP</td><td>69.2</td><td>89.9</td><td>96.1</td><td>81.5</td><td>96.9</td><td>99.1</td><td>63.0</td><td>86.6</td><td>92.9</td><td>83.5</td><td>97.3</td><td>99.2</td>
    </tr>
</table>
<br>

**Zero-shot Image Classification**:
<table border="1" width="150%">
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


# 开始用起来！
## 安装要求
开始本项目前，需先检查是否满足下列环境配置要求:

* python >= 3.6.4
* pytorch >= 1.8.0 (with torchvision >= 0.9.0)
* CUDA Version >= 10.2

运行下列命令即可安装本项目所需的三方库。

```bash
pip install -r requirements.txt
```

## API快速上手
下面提供一段简单的代码示例说明如何使用中文CLIP的API。开始使用前，请先安装cn_clip：
```bash
# 通过pip安装
pip install cn_clip

# 或者从源代码安装
cd Chinese-CLIP
pip install -e .
```
安装成功后，即可通过如下方式轻松调用API，提取图文特征并计算相似度：
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
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    image_features /= image_features.norm(dim=-1, keepdim=True) 
    text_features /= text_features.norm(dim=-1, keepdim=True)    

    logits_per_image, logits_per_text = model.get_similarity(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
```
如果你不满足于仅仅使用API，欢迎继续阅读本文档，了解如何使用我们的项目进行CLIP模型的训练和测试。
<br><br>


# 教程
下文将包括[跨模态检索教程](#跨模态检索)（包含finetune和inference，及KNN计算等）以及[零样本图像分类教程](#零样本图像分类)。

## 跨模态检索
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
    ├── Flickr30k-CN/
    └── .../          # 更多自定义数据集...
```

### 准备工作
这里我们提供预训练模型参数的下载方式，以及进行finetune前对数据进行的预处理过程

#### 预训练CKPT

请参考前文[模型规模 & 下载链接](#model_card)部分，下载对应模型ckpt。推荐将下载的ckpt文件存放于`${DATAPATH}/pretrained_weights/`目录下。

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
对于测试集只有文本，不知道图文对匹配关系的情况，每行的`image_ids`字段处理为空列表即可，即`"image_ids": []`。

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

为了降低上手难度，我们也提供了按上述步骤预处理好的MUGE数据（[下载链接](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/MUGE.zip)）和Flickr30K-CN数据（[下载链接](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/Flickr30k-CN.zip)）压缩包，直接下载解压并放置于`${DATAPATH}/datasets/`目录下即可。

### 模型finetune

在此我们介绍训练的步骤，方便其他用户了解模型细节，使用我们提供的中文CLIP预训练模型进行finetune。基于MUGE和Flickr30K-CN两个下游检索数据集，我们提供了训练样例脚本`run_scripts/muge_finetune_vit-b-16_rbt-base.sh`和`run_scripts/flickr30k_finetune_vit-b-16_rbt-base.sh`。<b>运行脚本同时支持单机和多机分布式训练，请在运行前，先根据脚本开头的指引注释，填写好分布式相关配置，之后运行如下命令即可开始训练（多机训练请在各机器上都运行命令）。对于显存不足的情况，可以考虑激活配置项中的[重计算策略](#checkpointing)。</b>训练产生的log和模型ckpt文件，会自动保存在用户指定的目录下：

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
  + `vision-model`: 指定视觉backbone, 从 `["ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"]`选择。
  + `text-model`: 指定文本backbone, 从 `["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"]`选择。
  + `context-length`: 文本输入序列长度。
  + `warmup`: warmup步数。
  + `batch-size`: 训练时单卡batch-size。（请保证`训练样本总数 > batch-size * GPU数`，至少满足1个训练batch）
  + `lr`: 学习率。
  + `wd`: weight decay。
  + `max-steps`: 训练步数，也可通过`max-epochs`指定训练轮数。
  + `freeze-vision`: 是否freeze视觉backbone。
  + `use-augment`: 是否使用[AutoAugment](https://arxiv.org/abs/1805.09501)对图片进行数据增强。
  + `valid-batch-size`: 验证时单机batch-size。（请保证`验证集样本总数 > batch-size * GPU数`，至少满足1个验证batch）
  + `valid-step-interval`和`valid-epoch-interval`: 验证step/epoch频率，指定为-1时则在训练中不进行验证。
  + `grad-checkpointing`: <span id="checkpointing"></span>使用[重计算策略](https://pytorch.org/docs/stable/checkpoint.html)，在前向过程中不保存中间结果，以训练时间换取更小的显存开销，适用于显存不足的情况。（`store_true`参数，直接在脚本中加上`--grad-checkpointing`即可，目前要求Pytorch>1.8.0）
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
    --context-length=52 \
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

对于小规模的学术检索数据集，我们提供一个简单的KNN检索实现，便于计算文到图、图到文检索的top-k召回结果（tips：如想仿照我们在项目中搭建检索demo，建议基于中文CLIP模型产出图文特征后，结合开源工程框架[clip-retrieval](https://github.com/rom1504/clip-retrieval)搭建前后端服务。）

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
<br>


## 零样本图像分类
本部分介绍如何使用Chinese CLIP实现零样本图像分类，以零样本图像分类Benchmark ELEVATER中的数据集为例。更多关于该benchmark的详情请点击[链接](https://eval.ai/web/challenges/challenge-page/1832/overview)。
<br>

### 准备工作
首先将数据按照如下格式进行准备。由于零样本图像分类仅需测试，因此只需要准备好测试集：
```
${DATAPATH}
└── datasets
    └── ${dataset_name}
        ├── label_cn.txt
        └── test
	    ├── 000 # label id，如label个数大于10，则将其向左补零到3位数保证字典序
	    │   ├── image_0003.jpg # 图片样本，命名无特殊要求
	    │   ├── image_0005.jpg
	    │   ├── ...
	    ├── 001
	    │   ├── image_0001.jpg
	    │   ├── image_0002.jpg
	    │   ├── ...
	    └── 002
	        ├── image_0003.jpg
	        ├── image_0005.jpg
	        └── ...
	    ...
	
```
测试集保证test文件夹内数据按照label对应的id进行划分，并保证id为字典序（10以上的多位数，需向左补零`label.zfill(3)`, 如001，002等）。`label_cn.txt`为数据标签，每行一个标签名，如下所示：
```
手风琴
飞机
锚
...
```
每行的标签对应的label id为`行号-1`，如第1行的标签的id为0，第二行的标签的id为1。如果标签总数大于10，则统一向左补零到3位数，比如标签个数为100，标签id则为`000-099`。用户需为每个label id生成对应的文件夹，并将标注该label的样本放入其中。我们以FGVC为样例，请点击[链接](https://shuangqing-multimodal.oss-cn-zhangjiakou.aliyuncs.com/cvinw/classification_organized/fgvc-aircraft-2013b-variants102-example.zip)下载。
<br>

### 预测和评估
我们准备了预测脚本，请查看`run_scripts/zeroshot_eval.sh`。运行命令例子如下：
```bash
bash run_scripts/zeroshot_eval.sh 0 \
    ${DATAPATH} ${dataset_name} \
    ${vision_model} ${text_model} \
    ${ckpt_path}
```
其中第一个入参`0`为GPU id，`vision_model`为指定模型类型，选项包括`["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "RN50", "ViT-H-14"]`，而`text_model`包括`["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"]`，`ckpt_path`即为模型ckpt的路径。

返回结果会打印top-1的准确率。同时，程序还会存下一个json文件用于提交ELEVATER用，json文件内容如下所示：
```json
{"model_name": "CN-CLIP-ViT-B-16", "dataset_name": "fgvc-aircraft-2013b-variants102", "num_trainable_params": 0, "num_params": 188262913, "num_visual_params": 86192640, "num_backbone_params": 188262913 "n_shot": 0, "rnd_seeds": [0], "predictions": "prediction probability tensor [size: (1, 10000, 101)]"}
```
其中包括模型名`model_name`、数据集名称`dataset_name`、总参数量`num_params`、视觉塔的参数量`num_visual_params`等模型的meta信息，以及模型输出结果，即模型的预测概率tensor，size为`[1, 样本数, 标签个数]`。
<br><br><br>


# 引用
如果觉得本项目好用，希望能给我们提个star并分享给身边的用户，欢迎给相关工作citation，感谢支持！

```
@article{chinese-clip,
  title={Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese},
  author={Yang, An and Pan, Junshu and Lin, Junyang and Men, Rui and Zhang, Yichang and Zhou, Jingren and Zhou, Chang},
  journal={arXiv preprint arXiv:2211.01335},
  year={2022}
}
```
