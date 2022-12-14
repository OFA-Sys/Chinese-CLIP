[**中文说明**](zeroshot_dataset.md) | [**English**](zeroshot_dataset_en.md)

# Zero-shot Image Classification Datasets

The collection of dataset is the Chinese version of the Image Classification in the Wild in the [ELEVATER Benchmark](https://eval.ai/web/challenges/challenge-page/1832). It consists of 20 datasets, including Caltech-101, CIFAR-10, CIFAR-100, MNIST, etc. We provide our organized datasets, which enable direct usage of our codes on the datasets. 

Download link: [Click here](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/datasets/ELEVATER_all.zip)

## Notes
We have organized 20 datasets into 20 directories, and zipped and uploaded them. Users can click the link above to download all the datasets. After unzipping the `ELEVATER_all.zip`, you will get the zipped files of each dataset in ELEVATER. Once again after unzipping the dataset zipfile, you will get the dataset folder which contains the following structure:
```
${dataset_name}
├── index.json  # Some datasets contain this file，which only serves for the submission to the ELEVATER benchmark
├── label_cn.txt  # File of Chinese labels，where the text in each line refers to the label name
├── label.txt  # File of English labels，where the text in each line refers to the label name
├── test/
│   ├── 000/
│   ├── 001/
│   └── 002/
└── train/
    ├── 000/
    ├── 001/
    └── 002/
```
`${dataset_name}` refers to the directory of each dataset, where there are two directories named `train` and `test`. Each directory contains sub-directories named with id, which refers to a category. Additionally, there are 2 files, namely the Chinese label file`label_cn.txt`, the English label file `label.txt`. Note that:

* When the number of labels is no larger than 10, e,g., 10, the ids are [0-9]
* When the number of labels is larger than 10, e.g., 100, the ids are [000-099]，which are left padded with 0 to 3-digit numbers. This serves for alphabetic order. 
* Each id refers to the label name in the ${id}-th line of the label file (0-index). For example, `0` refers to the label name in the 0-th line, and `099` refers to the label name in the 99-th line

The sub-directories of training and test sets are alphatically ordered for the reason that we use `torchvision.dataset`, which requires to organize data to sub-directories based on their labels and alphabetically ordered. 

There are two label files for Chinese and English. We only use `label_cn.txt` in our experiments, and `label.txt` is for reference only. Each line contains a label name. The example is shown below:
```
飞机
汽车
……
```

`index.json` only serves for the submission of the ELEVATER benchmark, and not every dataset contains this file. The existence of this file is due to the specified order of test data. To make your submission available, you need to add ` index.json` in your command. 
