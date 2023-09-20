[**中文说明**](distillation.md) | [**English**](distillation_En.md)

# Improving Chinese-CLIP Image Retrieval Ability Using Knowledge Distillation

Here we provide an example of knowledge distillation for Chinese-CLIP fine-tuning training, based on [ModelScope](https://github.com/modelscope/modelscope) model library. By using knowledge distillation, smaller Chinese-CLIP models (with better inference speed) can learn from larger models (including larger Chinese-CLIP or other image embedding models on ModelScope) to further improve the image-to-image retrieval ability. The Teacher models used are all from [ModelScope](https://github.com/modelscope/modelscope). Currently, all the Chinese-CLIP have been supported on ModelScope.

## Environmental Preparation

+ Nvidia GPUs **with Turning, Ampere, Ada or Hopper architecture** (such as H100, A100, RTX 3090, T4, and RTX 2080). Please refer to [this document](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) for the corresponding GPUs of each Nvidia architecture.
+ CUDA 11.4 and above.
+ PyTorch 1.12 and above.
+ **ModelScope**：Install ModelScope by executing `pip install modelscope`.
+ Other dependencies as required in [requirements.txt](requirements.txt).

## Use it in Chinese-CLIP!
It is not complicated to apply knowledge distillation to the image side in Chinese-CLIP finetune. Just add the `--distllation` configuration item to the sh script of finetune.
Then fill in the name of the Teacher model to be used in the configuration item `--teacher-model-name`. The currently supported Teacher models include the following four ModelScope-supported models.
<table border="1" width="120%">
    <tr align="center">
        <td><b>Teacher model</b></td><td><b>Model Info</b></td>
    </tr>  
	<tr align="center">
        <td>damo/multi-modal_clip-vit-huge-patch14_zh</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_clip-vit-huge-patch14_zh/summary">CLIP model-Chinese-general field-huge</a></td>
    </tr>
	<tr align="center">
        <td>damo/multi-modal_clip-vit-large-patch14_zh</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_clip-vit-large-patch14_zh/summary">CLIP model-Chinese-general field-large</a></td>
    </tr>	    
    </tr>
	<tr align="center">
        <td>damo/multi-modal_team-vit-large-patch14_multi-modal-similarity</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_team-vit-large-patch14_multi-modal-similarity/summary">TEAM image-text retrieval model-Chinese-large</a></td>
    </tr>  
	<tr align="center">
        <td>damo/multi-modal_rleg-vit-large-patch14</td><td><a href="https://www.modelscope.cn/models/damo/multi-modal_rleg-vit-large-patch14/summary">RLEG Generative Multimodal Representation Model-English-large</a></td>
</table>
<br>

Finally, fill in the weight of the distillation loss in the configuration item `--kd_loss_weight`, the default value is 0.5.

The configuration items are defined as follows:
+ `distllation`: Whether to enable knowledge distillation to fine-tune the image side of the model.
+ `teacher-model-name`: Specify the Teacher model to use. Currently supports the above four Teacher models, such as filling in `damo/multi-modal_team-vit-large-patch14_multi-modal-similarity`.
+ `kd_loss_weight` (optional): Distillation loss weight, default value is 0.5.

We provide a sample script `run_scripts/muge_finetune_vit-b-16_rbt-base_distllation.sh`, we take the `TEAM image-text retrieval model-Chinese-large` as Teacher model.

## Effect verification
Image retrieval Top10 results of our model (finetune+distillation) v.s. pre-trained model v.s. finetune model. The image in the upper left corner is used as a query, and the search results are in order from Top1 to Top10 on the right. The support data set in this experiment has 100,000 e-commerce data (including shoes, clothes, pants, etc.).

Advantages of our approach:
+ Meet the basic requirements of the retrieval task: under the premise of ensuring the category similarity, the image similarity is well realized.
+ Good performance and fast speed: Through the distillation method, the base model has a retrieval effect similar to that of the large model. And deployed to the CPU, the retrieval reasoning time is controlled within 100ms.

<p style="text-align: center;">
    <img src="examples/image_retrieval_result1.jpg" width="400" /><br>
    <img src="examples/image_retrieval_result3.jpg" width="400" /><br>
    <img src="examples/image_retrieval_result2.jpg" width="400" /><br>
</p>

## Quick Start
A solution of distillation have been launched on Alibaba Cloud [PAI-DSW Gallery](https://gallery.pai-ml.com/#/preview/deepLearning/cv/cn_clip_distillation). The corresponding Jupyter Notebook is provided in PAI-DSW Gallery to support users to build exclusive search models using their own data.
