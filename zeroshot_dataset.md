[**中文说明**](zeroshot_dataset.md) | [**English**](zeroshot_dataset_en.md)

# 零样本图像分类数据集

本数据集为[ELEVATER Benchmark](https://eval.ai/web/challenges/challenge-page/1832)的图像分类基准的**中文版**，共包括20个图像分类数据集，包括Caltech-101、CIFAR-10、CIFAR-100、MNIST等。我们提供整理好的数据集，可以直接接入Chinese CLIP的代码进行零样本分类。

下载链接：[点击这里](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/ELEVATER_all.zip)

ImageNet的原始数据请自行到官网下载（可参考[此文档](https://gist.github.com/antoinebrl/7d00d5cb6c95ef194c737392ef7e476a)下载并将验证集转为ImageFolder格式），本项目仅提供[中文标签](http://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/ImageNet-1K/label_cn.txt)和[英文标签](http://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/ImageNet-1K/label.txt)。


## 数据集说明
我们将20个数据集分别置于20个文件夹中，统一打包上传，用户通过点击上述链接即可下载全部数据。`ELEVATER_all.zip`解压后，将得到每个数据集的zip压缩包。选择对应的压缩包再次解压后，每个文件夹的内容如下所示：
```
${dataset_name}
├── index.json  # 个别数据集包含这个文件，仅用于提交ELEVATER benchmark
├── label_cn.txt  # 中文标签名文件，每一行一个类别名
├── label.txt  # 英文标签名文件，每一行一个类别名
├── test/
│   ├── 000/
│   ├── 001/
│   └── 002/
└── train/
    ├── 000/
    ├── 001/
    └── 002/
```
`${dataset_name}`表示每个数据集的文件夹路径，如`cifar-100`，里面包括`train`和`test`两个文件夹，每个文件夹包含了以id编号命名的文件夹，分别代表每一个类别。另外还包含3个文件，分别为中文标签名文件`label_cn.txt`和英文标签名文件`label.txt`。其中：

* 类别数在10个及以下的情况下，如10，类别的id分别为[0-9]
* 类别数在10个以上的情况下，如100，类别的id分别为[000-099]，即向左补零到3位数。这是为了保证我们的id是以字典序进行排序
* 每个id对应的类别标签名为标签文件中的第${id}行（0-index），如`0`即对应标签文件中的第0行的类别名，`099`对应的是标签文件的第99行类别名。

训练和测试集文件夹内包含的子文件夹用字典序排序的原因是因为我们的代码使用了torchvision的dataset，默认文件夹内数据按照类别归类子文件夹，按照文件名以字典序排序。

标签文件包含中文版和原版两个文件，我们的代码仅需使用`label_cn.txt`，`label.txt`仅供参考。文件内容为每一行1个类别名，示例如下：
```
飞机
汽车
……
```

`index.json`仅用于提交ELEVATER benchmark使用，且并非每个数据集都包含此文件。该文件的原因是ELEVATER官方评测部分数据集的测试集样本顺序经过调整，如需保证提交结果正常需要调整样本顺序。如遇到数据集包含此文件，则可在测试运行命令中加上` index.json`即可。

类似地，如您自行准备ImageNet数据，请将上述中文和英文标签文件放入`${dataset_name}`，并在其中创建相应文件夹，如`train`和`test`，将图片按照类别归档并放入对应文件夹，并保证其按字典序排序，如`000-999`，实现的文件结构和上述示例保持一致，即可实现零样本分类的数据准备。
