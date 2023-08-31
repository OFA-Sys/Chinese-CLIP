[**中文说明**](distillation.md) | [**English**](distillation_En.md)

# Improving Chinese-CLIP Image Retrieval Ability Using Knowledge Distillation

Chinese-CLIP combines knowledge distillation for fine-tuning training to further improve the image retrieval (image2image) ability of ChineseClip. The Teacher models used are all from [ModelScope](https://github.com/modelscope/modelscope).

## Environmental Preparation

+ Nvidia GPUs **with Turning, Ampere, Ada or Hopper architecture** (such as H100, A100, RTX 3090, T4, and RTX 2080). Please refer to [this document](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) for the corresponding GPUs of each Nvidia architecture.
+ CUDA 11.4 and above.
+ PyTorch 1.12 and above.
+ **ModelScope**：Install FlashAttention by executing `pip install modelscope`.
+ Other dependencies as required in [requirements.txt](requirements.txt).

## Use it in Chinese-CLIP!
It is not complicated to apply knowledge distillation to the image side in Chinese-CLIP finetune. Just add the `--distllation` configuration item to the sh script of finetune.
Then fill in the name of the Teacher model to be used in the configuration item `--teacher-model-name`. The currently supported Teacher modes include the following four.
<table border="1" width="120%">
    <tr align="center">
        <td><b>Teacher model</b></td><td><b>模型介绍</b></td>
    </tr>
	<tr align="center">
        <td>damo/multi-modal_team-vit-large-patch14_multi-modal-similarity</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_team-vit-large-patch14_multi-modal-similarity/summary">TEAM image-text retrieval model-Chinese-large</a></td>
    </tr>  
	<tr align="center">
        <td>damo/multi-modal_rleg-vit-large-patch14</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_rleg-vit-large-patch14/summary">RLEG Generative Multimodal Representation Model-English-large
</a></td>
    </tr>  
	<tr align="center">
        <td>damo/multi-modal_clip-vit-huge-patch14_zh</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_clip-vit-huge-patch14_zh/summary">CLIP model-Chinese-general field-huge</a></td>
    </tr>
	<tr align="center">
        <td>damo/multi-modal_clip-vit-large-patch14_zh</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_clip-vit-large-patch14_zh/summary">CLIP model-Chinese-general field-large</a></td>
    </tr>
</table>
<br>

Finally, fill in the weight of the distillation loss in the configuration item `--kd_loss_weight`, the default value is 0.5.

The configuration items are defined as follows:
+ `distllation`: Whether to enable knowledge distillation to fine-tune the image side of the model.
+ `teacher-model-name`: Specify the Teacher model to use. Currently supports the above four Teacher models, such as filling in `damo/multi-modal_team-vit-large-patch14_multi-modal-similarity`.
+ `kd_loss_weight` (optional): Distillation loss weight, default value is 0.5.

We provide a sample script `run_scripts/muge_finetune_vit-b-16_rbt-base_distllation.sh`.

## Todo
The Jupyter Notebook of related solutions will be launched on the Alibaba Cloud official website.