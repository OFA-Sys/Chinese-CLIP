[**中文说明**](flash_attention.md) | [**English**](flash_attention_En.md)

# 使用FlashAttention加速Chinese-CLIP

Chinese-CLIP训练现已支持通过[FlashAttention](https://github.com/HazyResearch/flash-attention)加速训练进程。

## 环境准备

+ **Turing**、**Ampere**、**Ada**、**Hopper**架构的Nvidia GPU显卡（如H100、A100、RTX 3090、T4、RTX 2080），Nvidia各架构对应显卡型号可参见[此文档表格](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)。
+ CUDA 11.4及以上版本。
+ Pytorch 1.12及以上版本。
+ **FlashAttention**：通过执行`pip install flash-attn`安装FlashAttention。

更多信息可参见[FlashAttention项目仓库](https://github.com/HazyResearch/flash-attention)。

## 在Chinese-CLIP中用起来！

在Chinese-CLIP finetune中应用FlashAttention非常简单，只需要在finetune的sh脚本中加入`--use-flash-attention`配置项即可。我们提供了样例脚本`run_scripts/muge_finetune_vit-b-16_rbt-base_flashattn.sh`。


## 训练速度和显存占用对比

启用FlashAttention可在不影响效果的条件下为Chinese-CLIP的finetune过程显著提速以及降低显存占用。我们的实验在一台8卡A100 GPU（80GB显存）机器进行，FlashAttention 0.2.8，Pytorch 1.10.1。

我们分别列出finetune过程中，相同batch size下启用FlashAttention前后每个规模模型的FP16精度finetune的batch time和显存占用对比，可以看到启用FlashAttention后，训练速度有所提升，也更加节约显存。对于更大规模模型的训练速度提升和显存占用降低更为显著。

<table border="1" width="120%">
    <tr align="center">
        <th></th><th colspan="4">Batch Time</th>
    </tr>
    <th>单位: 秒/it</th><th>Batch size</th><th>w/o FlashAttention</th><th>w/ FlashAttention</th><th>Speedup</th>
    </tr>
    <tr align="center">
        <td width="120%">CN-CLIP<sub>RN50</sub></td><td>1200*8</td><td>1.710</td><td>1.680</td><td>1.02×</td>
    </tr>  
    <tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>450*8</td><td>1.477</td><td>0.960</td><td>1.54×</td>
    </tr>  
    <tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>128*8</td><td>1.293</td><td>0.785</td><td>1.65×</td>
    </tr>
    <tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14@336px</sub></td><td>40*8</td><td>1.397</td><td>0.587</td><td>2.38×</td>
    </tr>
    <tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-H/14</sub></td><td>64*8</td><td>1.265</td><td>0.845</td><td>1.50×</td>
    </tr>  
</table>
<br>

<table border="1" width="120%">
    <tr align="center">
        <th></th><th colspan="4">显存</th>
    </tr>
    <th>单位: GB</th><th>Batch size</th><th>w/o FlashAttention</th><th>w/ FlashAttention</th>
    </tr>
    <tr align="center">
        <td width="120%">CN-CLIP<sub>RN50</sub></td><td>1200*8</td><td>79</td><td>75</td>
    </tr>  
    <tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>450*8</td><td>80</td><td>56</td>
    </tr>  
    <tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>128*8</td><td>77</td><td>50</td>
    </tr>
    <tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14@336px</sub></td><td>40*8</td><td>78</td><td>37</td>
    </tr>
    <tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-H/14</sub></td><td>64*8</td><td>76</td><td>57</td>
    </tr>  
</table>
<br>
