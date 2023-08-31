[**中文说明**](distillation.md) | [**English**](distillation_En.md)

# 使用知识蒸馏提升Chinese-CLIP图像检索能力

Chinese-CLIP结合知识蒸馏进行微调训练，进一步提升ChineseClip的图像检索(image2image)能力。使用的Teacher model全都来自[ModelScope](https://github.com/modelscope/modelscope)。

## 环境准备

+ **Turing**、**Ampere**、**Ada**、**Hopper**架构的Nvidia GPU显卡（如H100、A100、RTX 3090、T4、RTX 2080），Nvidia各架构对应显卡型号可参见[此文档表格](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)。
+ CUDA 11.4及以上版本。
+ Pytorch 1.12及以上版本。
+ [requirements.txt](requirements.txt)要求的其他依赖项
+ **ModelScope**：通过执行`pip install modelscope`安装ModelScope。

## 在Chinese-CLIP中用起来！

在Chinese-CLIP finetune中对于图像端应用知识蒸馏并不复杂。只需要在finetune的sh脚本中加入`--distllation`配置项。
然后在配置项`--teacher-model-name`填入所要使用的Teacher model名称。现在支持的Teacher mode包括以下四种。
<table border="1" width="120%">
    <tr align="center">
        <td><b>Teacher model</b></td><td><b>模型介绍</b></td>
    </tr>
	<tr align="center">
        <td>damo/multi-modal_team-vit-large-patch14_multi-modal-similarity</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_team-vit-large-patch14_multi-modal-similarity/summary">TEAM图文检索模型-中文-large</a></td>
    </tr>  
	<tr align="center">
        <td>damo/multi-modal_rleg-vit-large-patch14</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_rleg-vit-large-patch14/summary">RLEG生成式多模态表征模型-英文-large
</a></td>
    </tr>  
	<tr align="center">
        <td>damo/multi-modal_clip-vit-huge-patch14_zh</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_clip-vit-huge-patch14_zh/summary">CLIP模型-中文-通用领域-huge</a></td>
    </tr>
	<tr align="center">
        <td>damo/multi-modal_clip-vit-large-patch14_zh</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_clip-vit-large-patch14_zh/summary">CLIP模型-中文-通用领域-large</a></td>
    </tr>
</table>
<br>

最后在配置项`--kd_loss_weight`填入蒸馏损失的权值，默认值是0.5。


其中各配置项定义如下：
+ `distllation`: 是否启用知识蒸馏微调模型图像端。
+ `teacher-model-name`: 指定使用的Teacher model。目前支持以上四个Teacher model，如填入`damo/multi-modal_team-vit-large-patch14_multi-modal-similarity`。
+ `kd_loss_weight`（可选）: 蒸馏损失的权值，默认值是0.5。

我们提供了样例脚本`run_scripts/muge_finetune_vit-b-16_rbt-base_distllation.sh`。

## Todo
将会在阿里云官网上线相关的解决方案的Jupyter Notebook。