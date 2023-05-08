[**中文说明**](flash_attention.md) | [**English**](flash_attention_En.md)

# Accelerate Chinese-CLIP with FlashAttention

Chinese-CLIP now supports the acceleration of training process through [FlashAttention](https://github.com/HazyResearch/flash-attention).

## Environmental Preparation

+ Nvidia GPUs **with Turning, Ampere, Ada or Hopper architecture** (such as H100, A100, RTX 3090, T4, and RTX 2080). Please refer to [this document](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) for the corresponding GPUs of each Nvidia architecture.
+ CUDA 11.4 and above.
+ PyTorch 1.12 and above.
+ **FlashAttention**：Install FlashAttention by executing `pip install flash-attn`.

Please refer to the [FlashAttention project repository](https://github.com/HazyResearch/flash-attention) for more information.

## Use it in Chinese-CLIP!

Applying FlashAttention to the finetune process of Chinese-CLIP is very simple, just add `--use-flash-attention` to the sh script of finetune. We provide the sample script `run_scripts/muge_finetune_vit-b-16_rbt-base_flashattn.sh`.


## Training Speed and Memory Usage Comparison

Enabling FlashAttention can significantly speed up the finetune process and reduce the memory usage of Chinese-CLIP without affecting the precision. Our experiments are conducted on an 8-card A100 GPU (80GB memory) machine，FlashAttention 0.2.8，Pytorch 1.10.1.

We present the comparison of the batch time and memory usage of FP16 precision finetune for each scale model. The improvement in training speed and reduction in memory usage are more significant for larger models.

<table border="1" width="120%">
    <tr align="center">
        <th></th><th colspan="4">Batch Time</th>
    </tr>
    <th>Unit: s/it</th><th>Batch size</th><th>w/o FlashAttention</th><th>w/ FlashAttention</th><th>Speedup</th>
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
        <th></th><th colspan="4">Memory</th>
    </tr>
    <th>Unit: GB</th><th>Batch size</th><th>w/o FlashAttention</th><th>w/ FlashAttention</th>
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
