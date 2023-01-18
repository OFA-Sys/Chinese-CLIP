[**中文说明**](deployment.md) | [**English**](deployment_En.md)

# Chinese-CLIP Model Deployment: ONNX & TensorRT Format Conversion

Our latest Chinese-CLIP code supports the conversion of Pytorch models of all scales into [ONNX](https://onnx.ai/) or [TensorRT](https://developer.nvidia.com/tensorrt) formats, thereby **[improving the inference speed of feature calculation](#speed-comparison-results)** compared with the original Pytorch models without affecting the downstream task effect of feature extraction. Below we give the whole process of preparing the FP16 Chinese-CLIP models in ONNX and TensorRT formats on GPU (and also give the [download links](#tensorrt_download) of Chinese-CLIP pretraining TensorRT models), and attach the comparison of model effect and inference speed, so that you can take advantage of the advantages of ONNX and TensorRT library in inference performance.

## Environmental Preparation

+ **GPU hardware requirements**: Please prepare Nvidia GPUs **with Volta architecture and above** (equipped with FP16 Tensor Core). Please refer to [this document](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) for the corresponding GPUs of each Nvidia architecture. Here we take T4 GPU as an example.
+ **CUDA**: [CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive) version 11.6 and above is recommended. We take version 11.6 as an example.
+ **CUDNN**: [CUDNN](https://developer.nvidia.com/rdp/cudnn-archive) version 8.6.0 and above is recommended. We take version 8.6.0 as an example. Please note that TensorRT and CUDNN have version correspondence, e.g. TensorRT 8.5.x must correspond to CUDNN 8.6.0, see the TensorRT version requirements for details.
+ **ONNX**: Please run `pip install onnx onnxruntime-gpu onnxmltools` to install. Note that when we convert the TensorRT model, we will follow the steps Pytorch → ONNX → TensorRT, so preparing the TensorRT model also requires installing the ONNX library first. Here we take onnx version 1.13.0, onnxruntime-gpu version 1.13.1, and onnxmltools version 1.11.1 as examples.
+ **TensorRT**: The recommended [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html#trt_8) version is 8.5.x. We use 8.5.2.2 as an example, using pip to install `pip install tensorrt==8.5.2.2`. For the CUDNN version corresponding to each TensorRT version, please refer to the "NVIDIA TensorRT Support Matrix" from this [documentation page]((https://docs.nvidia.com/deeplearning/tensorrt/archives/index.html#trt_8)).
+ **Pytorch**: Pytorch version 1.12.1 and above is recommended. We take version 1.12.1 as an example. (It is recommended to directly pip install 1.12.1 + cu116, and try not to use conda to install cudatoolkit, avoiding TensorRT errors due to CUDNN version changes. )
+ Other dependencies as required in [requirements.txt](requirements.txt).


## Converting and Running ONNX Models

### Converting Models

For code to convert a Pytorch checkpoint to ONNX format, see `cn_clip/deploy/pytorch_to_onnx.py`. Let's take the example of converting a ViT-B-16 size Chinese-CLIP pretrained model. You can run the following code (Please refer to the [Code Organization](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/README_En.md#code-organization) section of Readme to set `${DATAPATH}` and replace the script content below, using relative paths where possible. ) :

```bash
cd Chinese-CLIP/
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

# Please refer to the 'Code Organization' section in Readme to set `${DATAPATH}` and replace the script content below, using relative paths where possible: https://github.com/OFA-Sys/Chinese-CLIP/blob/master/README_En.md#code-organization
checkpoint_path=${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt # Specify the full path to the ckpt to be converted
mkdir -p ${DATAPATH}/deploy/ # Create output folders for ONNX models

python cn_clip/deploy/pytorch_to_onnx.py \
       --model-arch ViT-B-16 \
       --pytorch-ckpt-path ${checkpoint_path} \
       --save-onnx-path ${DATAPATH}/deploy/vit-b-16 \
       --convert-text --convert-vision
```

Each configuration item is defined as follows:

+ `model-arch`: Model size, options include`["RN50", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14"]`, details of each model can be found in [Readme](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/README_En.md#model-card). 
+ `pytorch-ckpt-path`: Specify the Pytorch model ckpt path, which in the code example above we specified as the pretrained ckpt path, or you can specify the location of your finetune ckpt. The parameters in ckpt need to correspond to the model size specified by `model-arch`.
+ `save-onnx-path`: Specifies the path (prefix) to the output ONNX format model. After the conversion is completed, the code will output the text-side and image-side ONNX format encoded model files, one for FP32 and one for FP16, which specifies the path prefix of the above output files.
+ `convert-text` and `convert-vision`: Specify whether to convert text-side and image-side models.
+ `context-length` (Optional): Specify the sequence length of the input received by the text-side ONNX model, defaulting to the 52 used in our pretraining ckpt.
+ `download-root` (Optional): If `pytorch-ckpt-path` is not specified, the code will automatically download the official Chinese-CLIP pretraining ckpt for conversion according to `model-arch`, which is stored in the specified directory `download-root`.

After running, the following log output will be obtained:
```
Finished PyTorch to ONNX conversion...
>>> The text FP32 ONNX model is saved at ${DATAPATH}/deploy/vit-b-16.txt.fp32.onnx
>>> The text FP16 ONNX model is saved at ${DATAPATH}/deploy/vit-b-16.txt.fp16.onnx with extra file ${DATAPATH}/deploy/vit-b-16.txt.fp16.onnx.extra_file
>>> The vision FP32 ONNX model is saved at ${DATAPATH}/deploy/vit-b-16.img.fp32.onnx
>>> The vision FP16 ONNX model is saved at ${DATAPATH}/deploy/vit-b-16.img.fp16.onnx with extra file ${DATAPATH}/deploy/vit-b-16.img.fp16.onnx.extra_file
```

After the above code is executed, we get the ViT-B-16 size Chinese-CLIP text-side and image-side ONNX models, which can be used to extract image and text features respectively. The paths to the output ONNX models are all prefixed with `save-onnx-path` at the time of running the script, followed by `.img`/`.txt`, `.fp16`/`.fp32`, and `.onnx`. We will subsequently use mainly ONNX models `vit-b-16.txt.fp16.onnx` and `vit-b-16.img.fp16.onnx` in FP16 format.

Notice that some of the ONNX model files also come with an extra_file, which is also part of the corresponding ONNX model. When using these ONNX models, the path to the extra_file is stored in the `.onnx` file (e.g. `${DATAPATH}/deploy/vit-b-16.txt.fp16.onnx.extra_file`) and the extra_file will be loaded according to this path, so please do not change the path and use relative path for `${DATAPATH}` when converting (e.g. `../datapath`) to avoid errors at runtime when the extra_file is not found.

### Run the Model

#### Extraction of Image-side Features
We use the following sample code in the `Chinese-CLIP/` directory to read the converted ViT-B-16 size ONNX image-side model `vit-b-16.img.fp16.onnx` and extract the features of the [Pikachu image](examples/pokemon.jpeg) in Readme. Note that the converted ONNX model only accepts inputs with a batch size of 1, i.e. only one input image is processed in one call.

```python
# Complete the necessary import (omitted below)
import onnxruntime
from PIL import Image
import numpy as np
import torch
import argparse
import cn_clip.clip as clip
from clip import load_from_name, available_models
from clip.utils import _MODELS, _MODEL_INFO, _download, available_models, create_model, image_transform

# Load ONNX image-side model（**Please replace ${DATAPATH} with the actual path**）
img_sess_options = onnxruntime.SessionOptions()
img_run_options = onnxruntime.RunOptions()
img_run_options.log_severity_level = 2
img_onnx_model_path="${DATAPATH}/deploy/vit-b-16.img.fp16.onnx"
img_session = onnxruntime.InferenceSession(img_onnx_model_path,
                                        sess_options=img_sess_options,
                                        providers=["CUDAExecutionProvider"])

# Preprocess images
model_arch = "ViT-B-16" # Here we use the ViT-B-16 size, other sizes please modify accordingly
preprocess = image_transform(_MODEL_INFO[model_arch]['input_resolution'])
# Example Pikachu image, Torch Tensor of [1, 3, resolution, resolution] size after preprocessing
image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0)

# Calculate image-side features with ONNX model
image_features = img_session.run(["unnorm_image_features"], {"image": image.cpu().numpy()})[0] # Unnormalized image features
image_features = torch.tensor(image_features)
image_features /= image_features.norm(dim=-1, keepdim=True) # Normalized Chinese-CLIP image features for downstream tasks
print(image_features.shape) # Torch Tensor shape: [1, feature dimension]
```

#### Extraction of Text-side Features

Similarly, we use the following code to complete the loading and feature calculation of the text-side ONNX model. As with the image-side model, the text-side ONNX model only accepts inputs with a batch size of 1, i.e., only one input text is processed in a single call.

```python
# Load ONNX text-side model（**Please replace ${DATAPATH} with the actual path**）
txt_sess_options = onnxruntime.SessionOptions()
txt_run_options = onnxruntime.RunOptions()
txt_run_options.log_severity_level = 2
txt_onnx_model_path="${DATAPATH}/deploy/vit-b-16.txt.fp16.onnx"
txt_session = onnxruntime.InferenceSession(txt_onnx_model_path,
                                        sess_options=txt_sess_options,
                                        providers=["CUDAExecutionProvider"])

# Tokenize the 4 input texts. The sequence length is specified to 52, which is the same as when converting the ONNX model (see the context_length in the conversion process).
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], context_length=52) 

# Calculate text-side features sequentially with ONNX model
text_features = []
for i in range(len(text)):
    one_text = np.expand_dims(text[i].cpu().numpy(),axis=0)
    text_feature = txt_session.run(["unnorm_text_features"], {"text":one_text})[0] # Unnormalized image features
    text_feature = torch.tensor(text_feature)
    text_features.append(text_feature)
text_features = torch.squeeze(torch.stack(text_features),dim=1) # 4 feature vectors stacked together
text_features = text_features / text_features.norm(dim=1, keepdim=True) # Normalized Chinese-CLIP text features for downstream tasks
print(text_features.shape) # Torch Tensor shape: [1, feature dimension]
```

#### Calculate Image & Text Similarity

Similarities are calculated from the normalized image and text features produced by the ONNX model after inner product and softmax (need to consider `logit_scale`), which is identical to the original Pytorch models, see the following code:

```python
# Inner product followed by softmax
# Note that in the inner product calculation, due to the concept of temperature during contrast learning training, you need to multiply the model logit_scale.exp(), our pretraining model logit_scale is 4.6052, so here multiply by 100.
# For your own ckpt, please load it using torch.load and then check ckpt['state_dict']['module.logit_scale'] or ckpt['state_dict']['logit_scale'].
logits_per_image = 100 * image_features @ text_features.t()
print(logits_per_image.softmax(dim=-1)) # Image & text similarity probabilities: [[1.2252e-03, 5.2874e-02, 6.7116e-04, 9.4523e-01]]
```

We can see that the output similarities given by the ONNX model are largely consistent with the results calculated in [API Use Case](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/README_En.md#api-use-case) section of Readme based on the same model in Pytorch, proving the correctness of the feature calculation of the ONNX model. However, **the feature calculation speed of ONNX model is more advantageous than that of Pytorch** (see [below](#Speed Comparison Results) for details).

## Converting and Running TensorRT Models

### Converting Models

Compared with ONNX models, TensorRT models have faster inference speed, and we provide converted Chinese-CLIP pretrained TensorRT image-side and text-side models ([download links](#tensorrt_download)). As mentioned above, we prepare the TensorRT format model, which is further transformed using the ONNX format model we just obtained. For the code to convert ONNX to TensorRT format, see `cn_clip/deploy/onnx_to_tensorrt.py`. Still using the ViT-B-16 size Chinese-CLIP as an example, using the ONNX models `vit-b-16.txt.fp16.onnx` and `vit-b-16.img.fp16.onnx`, run the following code in `Chinese-CLIP/`:

```bash
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip
# As above, please replace ${DATAPATH} according to the actual situation
python cn_clip/deploy/onnx_to_tensorrt.py \
       --model-arch ViT-B-16 \
       --convert-text \
       --text-onnx-path ${DATAPATH}/deploy/vit-b-16.txt.fp16.onnx \
       --convert-vision \
       --vision-onnx-path ${DATAPATH}/deploy/vit-b-16.img.fp16.onnx \
       --save-tensorrt-path ${DATAPATH}/deploy/vit-b-16 \
       --fp16
```

Each configuration item is defined as follows:

+ `model-arch`: Model size, options include`["RN50", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14"]`, details of each model can be found in [Readme](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/README_En.md#model-card). 
+ `convert-text` and `convert-vision`: Specify whether to convert text-side and image-side models.
+ `text-onnx-path`: Specify the text-side ONNX model file path, which needs to correspond to the model size specified by `model-arch`.
+ `vision-onnx-path`: Specify the image-side ONNX model file path, which needs to correspond to the model size specified by `model-arch`.
+ `save-tensorrt-path`: Specifies the path (prefix) to the output TensorRT format model. After the conversion, the code will output the model files encoded in TensorRT format for the text side and image side respectively, and this parameter specifies the path prefix of the above output files.
+ `fp16`: Specify models converted to FP16 precision TensorRT format.

The process takes a few minutes to ten minutes, depending on the model's size. After running, the following log output will be obtained:

```
Finished ONNX to TensorRT conversion...
>>> The text FP16 TensorRT model is saved at ${DATAPATH}/deploy/vit-b-16.txt.fp16.trt
>>> The vision FP16 TensorRT model is saved at ${DATAPATH}/deploy/vit-b-16.img.fp16.trt
```

After the execution of the above sample code, we obtained ViT-B-16 size Chinese-CLIP text-side and image-side TensorRT format models using ONNX models, which can be used to extract image & text features. The paths to the output TensorRT models are all prefixed with `save-tensorrt-path` when running the script, followed by `.img`/`.txt`, `.fp16`, and `.trt`. We use two output files `vit-b-16.txt.fp16.trt` and `vit-b-16.img.fp16.trt`.

**For each size of Chinese-CLIP pretrained models, we provide converted TensorRT image-side and text-side models (based on TensorRT version 8.5.2.2)**, which can be downloaded as follows<span id="tensorrt_download"></span>:

<table border="1" width="120%">
    <tr align="center">
        <td><b>Model Size</b></td><td><b>TensorRT Image-side Model</b></td><td><b>TensorRT Text-side Model</b></td>
    </tr>
	<tr align="center">
        <td>CN-CLIP<sub>RN50</sub></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/rn50.img.fp16.trt">Download</a></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/rn50.img.fp16.trt">Download</a></td>
    </tr>  
	<tr align="center">
        <td>CN-CLIP<sub>ViT-B/16</sub></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vit-b-16.img.fp16.trt">Download</a></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vit-b-16.txt.fp16.trt">Download</a></td>
    </tr>  
	<tr align="center">
        <td>CN-CLIP<sub>ViT-L/14</sub></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vit-l-14.img.fp16.trt">Download</a></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vit-l-14.txt.fp16.trt">Download</a></td>
    </tr>
	<tr align="center">
        <td>CN-CLIP<sub>ViT-L/14@336px</sub></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vit-l-14-336.img.fp16.trt">Download</a></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vit-l-14-336.txt.fp16.trt">Download</a></td>
    </tr>
	<tr align="center">
        <td>CN-CLIP<sub>ViT-H/14</sub></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vit-h-14.img.fp16.trt">Download</a></td><td><a href="https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vit-h-14.txt.fp16.trt">Download</a></td>
    </tr>  
</table>

<br>

Just download it and place it directly under directory `${DATAPATH}/deploy/`.

### Running Models

When running the TensorRT model, if the conversion and running are not in the same environment, please note that the TensorRT library version of the environment running the model is consistent with the conversion to avoid errors.

#### Extraction of Image-side Features

Similar to the process of ONNX models, we use the following sample code in the `Chinese-CLIP/` directory to read the converted ViT-B-16 size TensorRT image-side model `vit-b-16.img.fp16.trt` and extract the features of the [Pikachu image](examples/pokemon.jpeg) in Readme. The converted TensorRT model here also accepts only inputs with a batch size of 1, i.e. only one input image is processed in one call.

```python
# Complete the necessary import (omitted below)
from cn_clip.deploy.tensorrt_utils import TensorRTModel
from PIL import Image
import numpy as np
import torch
import argparse
import cn_clip.clip as clip
from clip import load_from_name, available_models
from clip.utils import _MODELS, _MODEL_INFO, _download, available_models, create_model, image_transform

# Load ONNX image-side model（**Please replace ${DATAPATH} with the actual path**）
img_trt_model_path="${DATAPATH}/deploy/vit-b-16.img.fp16.trt"
img_trt_model = TensorRTModel(img_trt_model_path)

# Preprocess images
model_arch = "ViT-B-16" # Here we use the ViT-B-16 size, other sizes please modify accordingly
preprocess = image_transform(_MODEL_INFO[model_arch]['input_resolution'])
# Example Pikachu image, Torch Tensor of [1, 3, resolution, resolution] size after preprocessing
image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).cuda()

# Calculate image-side features with TensorRT model
image_features = img_trt_model(inputs={'image': image})['unnorm_image_features'] # Unnormalized image features
image_features /= image_features.norm(dim=-1, keepdim=True) # Normalized Chinese-CLIP image features for downstream tasks
print(image_features.shape) # Torch Tensor shape: [1, feature dimension]
```

#### Extraction of Text-side Features

Similarly, we use the following code to complete the loading and feature calculation of the text-side TensorRT model. As with the image-side model, the text-side TensorRT model only accepts inputs with a batch size of 1, i.e., only one input text is processed in a single call. The text sequence length accepted by TensorRT is consistent with the used ONNX model, see the context-length parameter during ONNX conversion.

```python
# Load TensorRT text-side model（**Please replace ${DATAPATH} with the actual path**）
txt_trt_model_path="${DATAPATH}/deploy/vit-b-16.txt.fp16.trt"
txt_trt_model = TensorRTModel(txt_trt_model_path)

# Tokenize the 4 input texts. The sequence length is specified to 52, which is the same as when converting the ONNX model (see the context_length in the ONNX conversion process).
text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], context_length=52).cuda()

# Calculate text-side features sequentially with TensorRT model
text_features = []
for i in range(len(text)):
    # Unnormalized image features
    text_feature = txt_trt_model(inputs={'text': torch.unsqueeze(text[i], dim=0)})['unnorm_text_features']
    text_features.append(text_feature)
text_features = torch.squeeze(torch.stack(text_features), dim=1) # 4 feature vectors stacked together
text_features = text_features / text_features.norm(dim=1, keepdim=True) # Normalized Chinese-CLIP text features for downstream tasks
print(text_features.shape) # Torch Tensor shape: [1, feature dimension]
```

#### Calculate Image & Text Similarity

Similarities are calculated from the normalized image and text features produced by the TensorRT model after inner product and softmax (need to consider `logit_scale`), which is identical to the original Pytorch models and ONNX models, see the following code:

```python
# Inner product followed by softmax
# Note that in the inner product calculation, due to the concept of temperature during contrast learning training, you need to multiply the model logit_scale.exp(), our pretraining model logit_scale is 4.6052, so here multiply by 100.
# For your own ckpt, please load it using torch.load and then check ckpt['state_dict']['module.logit_scale'] or ckpt['state_dict']['logit_scale'].
logits_per_image = 100 * image_features @ text_features.t()
print(logits_per_image.softmax(dim=-1)) # Image & text similarity probabilities: [[1.2475e-03, 5.3037e-02, 6.7583e-04, 9.4504e-01]]
```

We can see that the output similarities given by the TensorRT model are largely consistent with the results calculated in [API Use Case](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/README_En.md#api-use-case) section of Readme based on the same model in Pytorch and the results calculated by ONNX models above, proving the correctness of the feature calculation of the TensorRT model. However, the feature calculation speed of TensorRT model is **superior to both of the previous two** (see [below](#Speed Comparison Results) for details).

## Comparison of Inference Speed

### Comparative Experimental Setup

Our experiments are conducted on a single T4 GPU (16GB memory) machine with 16 Intel Xeon (Skylake) Platinum 8163 (2.5GHz) CPU cores and 64GB memory. We use the above sample image and one of the candidate texts and perform 100 times of image and text feature extraction for both Pytorch, ONNX, and TensorRT models, taking the average time (ms). Taking the speed measurement of ViT-B-16 size model as an example, the code executed under `Chinese-CLIP/` is as follows:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

# Please replace ${DATAPATH} according to the actual situation
python3 cn_clip/deploy/speed_benchmark.py \
        --model-arch ViT-B-16 \
        --pytorch-ckpt ${DATAPATH}/pretrained_weights/clip_cn_vit-b-16.pt \
        --pytorch-precision fp16 \
        --onnx-image-model ${DATAPATH}/deploy/vit-b-16.img.fp16.onnx \
        --onnx-text-model ${DATAPATH}/deploy/vit-b-16.txt.fp16.onnx \
        --tensorrt-image-model ${DATAPATH}/deploy/vit-b-16.img.fp16.trt \
        --tensorrt-text-model ${DATAPATH}/deploy/vit-b-16.txt.fp16.trt
```

The following lines will be printed in the log output:

```
[Pytorch image inference speed (batch-size: 1):] mean=11.12ms, sd=0.05ms, min=11.00ms, max=11.32ms, median=11.11ms, 95p=11.20ms, 99p=11.30ms
[ONNX image inference speed (batch-size: 1):] mean=4.92ms, sd=0.04ms, min=4.82ms, max=5.01ms, median=4.92ms, 95p=4.98ms, 99p=5.00ms
[TensorRT image inference speed (batch-size: 1):] mean=3.58ms, sd=0.08ms, min=3.30ms, max=3.72ms, median=3.58ms, 95p=3.70ms, 99p=3.72ms

[Pytorch text inference speed (batch-size: 1):] mean=12.47ms, sd=0.07ms, min=12.32ms, max=12.64ms, median=12.48ms, 95p=12.57ms, 99p=12.61ms
[ONNX text inference speed (batch-size: 1):] mean=3.42ms, sd=0.44ms, min=2.96ms, max=3.89ms, median=3.45ms, 95p=3.87ms, 99p=3.88ms
[TensorRT text inference speed (batch-size: 1):] mean=1.54ms, sd=0.01ms, min=1.51ms, max=1.57ms, median=1.54ms, 95p=1.56ms, 99p=1.56ms
```

### Speed Comparison Results

We present a comparison of the FP16 precision inference time for each size of Pytorch, ONNX, and TensorRT models for an inference batch size of 1, and we can see that TensorRT has a particularly significant speedup for small-scale models.

<table border="1" width="120%">
    <tr align="center">
        <th>Unit: ms/sample</th><th colspan="3">Image Feature Extraction</th><th colspan="3">Text Feature Extraction</th>
    </tr>
    <tr align="center">
        <td>Models</td><td>Pytorch</td><td>ONNX</td><td>TensorRT</td><td>Pytorch</td><td>ONNX</td><td>TensorRT</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>RN50</sub></td><td>12.93</td><td>5.04</td><td><b>1.36</b></td><td>3.64</td><td>0.95</td><td><b>0.58</b></td>
    </tr>  
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-B/16</sub></td><td>11.12</td><td>4.92</td><td><b>3.58</b></td><td>12.47</td><td>3.42</td><td><b>1.54</b></td>
    </tr>  
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14</sub></td><td>21.19</td><td>17.10</td><td><b>13.08</b></td><td>12.45</td><td>3.48</td><td><b>1.52</b></td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-L/14@336px</sub></td><td>47.11</td><td>48.40</td><td><b>31.59</b></td><td>12.24</td><td>3.25</td><td><b>1.54</b></td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP<sub>ViT-H/14</sub></td><td>35.10</td><td>34.00</td><td><b>26.98</b></td><td>23.98</td><td>6.01</td><td><b>3.89</b></td>
    </tr>  
</table>
<br>

## Comparison of Downstream Tasks

We observe the zero-shot performance of the Pytorch, ONNX, and TensorRT FP16 models in the MUGE text-to-image retrieval task involved in the Chinese-CLIP experiments. As described in [Inference and Evaluation](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/README_En.md#inference-and-evaluation) section of Readme, the results of MUGE image & text retrieval evaluation are divided into 3 steps: image & text feature extraction, KNN retrieval, and Recall calculation. The image & text feature extraction scripts for ONNX and TensorRT models, please see `cn_clip/eval/extract_features_onnx.py` and `cn_clip/eval/extract_features_tensorrt.py` respectively, compared with `extract_features.py` used for original Pytorch feature extraction, only minor changes have been made. The scripts and processes used for the subsequent KNN and Recall calculations remain exactly the same.

The results of ViT-B-16 and ViT-H-14 scales of Chinese-CLIP are compared as follows:
<table border="1" width="100%">
    <tr align="center">
        <th>Setup</th><th colspan="4">ViT-B-16 Zero-shot</th><th colspan="4">ViT-H-14 Zero-shot</th>
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td><td>R@1</td><td>R@5</td><td>R@10</td><td>MR</td>
    </tr>
	<tr align="center">
        <td width="120%">Pytorch FP16</sub></td><td>52.1</td><td>76.7</td><td>84.4</td><td>71.1</td><td>63.0</td><td>84.1</td><td>89.2</td><td>78.8</td>
    </tr>  
	<tr align="center">
        <td width="120%">ONNX FP16</sub></td><td>52.0</td><td>76.8</td><td>84.3</td><td>71.1</td><td>63.1</td><td>84.1</td><td>89.0</td><td>78.8</td>
    </tr>
	<tr align="center">
        <td width="120%">TensorRT FP16</sub></td><td>52.0</td><td>76.8</td><td>84.2</td><td>71.0</td><td>63.1</td><td>84.2</td><td>89.1</td><td>78.8</td>
    </tr>
</table>
<br>

The results are basically the same, with a difference of ±0.2 within an acceptable range (the magnitude of error that can be caused by changing a machine), which proves the correctness of the conversion of the ONNX and TensorRT models.
