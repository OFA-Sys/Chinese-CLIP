import pandas as pd
import os
import base64
import json
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split

'''
original_dataset原始数据的路径文件夹，需修改为实际的路径
'''

#训练和验证集文本数据的文件
data1 = pd.read_csv('original_dataset/data1/ImageWordData.csv')
#训练和验证集图像数据的目录
data1_images_folder='original_dataset/data1/ImageData'

# 先将文本及对应图像id划分划分训练集和验证集
train_data, val_data = train_test_split(data1, test_size=0.2, random_state=42)

# 创建函数来处理数据集，使文本关联到其对应图像id的图像
def process_train_valid(data, images_folder, img_file, txt_file):
    with open(img_file, 'w') as f_img, open(txt_file, 'w') as f_txt:
        for index, row in data.iterrows():
            # 图片内容需要被编码为base64格式
            img_path = os.path.join(images_folder, row['image_id'])
            with open(img_path, 'rb') as f_img_file:
                img = Image.open(f_img_file)
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format)
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data).decode("utf-8")

            f_img.write(f"{row['image_id']}\t{base64_str}\n")

            # 文本内容和图片id需要被写入jsonl文件
            text_data = {"text_id": row["image_id"], "text": row["caption"], "image_ids": [row["image_id"]]}
            f_txt.write(json.dumps(text_data) + '\n')

# 处理训练集和验证集
# datasets/DatasetName为在Chinese-CLIP项目目录下新建的存放转换后数据集的文件夹
process_train_valid(train_data, data1_images_folder, 'Chinese-CLIP/datasets/DatasetName/train_imgs.tsv', 'Chinese-CLIP/datasets/DatasetName/train_texts.jsonl')
process_train_valid(val_data, data1_images_folder, 'Chinese-CLIP/datasets/DatasetName/valid_imgs.tsv', 'Chinese-CLIP/datasets/DatasetName/valid_texts.jsonl')



# 制作从文本到图像（Text_to_Image）检索时的，测试集。data2为Text_to_Image测试数据文件夹名
image_data2 = pd.read_csv('original_dataset/data2/image_data.csv')
word_test2 = pd.read_csv('original_dataset/data2/word_test.csv')
# 原始图像测试集目录
data2_images_folder='original_dataset/data2/ImageData'

# 处理Text_to_Image测试集
def process_text_to_image(image_data, images_folder, word_test, img_file, txt_file):
    with open(img_file, 'w') as f_img, open(txt_file, 'w') as f_txt:
        for index, row in image_data.iterrows():
            # 图片内容需要被编码为base64格式
            img_path = os.path.join(images_folder, row['image_id'])
            with open(img_path, 'rb') as f_img_file:
                img = Image.open(f_img_file)
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format)
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data).decode("utf-8")

            f_img.write(f"{row['image_id']}\t{base64_str}\n")

        for index, row in word_test.iterrows():
            # 文本内容和图片id需要被写入jsonl文件
            text_data = {"text_id": row["text_id"], "text": row["caption"], "image_ids": []}
            f_txt.write(json.dumps(text_data) + '\n')

# datasets/DatasetName为在Chinese-CLIP项目目录下新建的存放转换后数据集的文件夹
process_text_to_image(image_data2, data2_images_folder, word_test2, 'Chinese-CLIP/datasets/DatasetName/test2_imgs.tsv', 'Chinese-CLIP/datasets/DatasetName/test2_texts.jsonl')



# 制作从图像到文本（Image_to_Text）检索时的，测试集。data3为Image_to_Text测试数据文件夹名
image_test3 = pd.read_csv('original_dataset/data3/image_test.csv')
word_data3 = pd.read_csv('original_dataset/data3/word_data.csv')
# 原始图像测试集目录
data3_images_folder='original_dataset/data3/ImageData'

# 处理Image_to_Text测试集集
def process_image_to_text(image_data, images_folder, word_test, img_file, txt_file):
    with open(img_file, 'w') as f_img, open(txt_file, 'w') as f_txt:
        for index, row in image_data.iterrows():
            # 图片内容需要被编码为base64格式
            img_path = os.path.join(images_folder, row['image_id'])
            with open(img_path, 'rb') as f_img_file:
                img = Image.open(f_img_file)
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format)
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data).decode("utf-8")

            f_img.write(f"{row['image_id']}\t{base64_str}\n")

        for index, row in word_test.iterrows():
            # 文本内容和图片id需要被写入jsonl文件
            text_data = {"text_id": row["text_id"], "text": row["caption"], "image_ids": []}
            f_txt.write(json.dumps(text_data) + '\n')

# datasets/DatasetName为在Chinese-CLIP项目目录下新建的存放转换后数据集的文件夹
process_image_to_text(image_test3, data3_images_folder, word_data3, 'Chinese-CLIP/datasets/DatasetName/test3_imgs.tsv', 'Chinese-CLIP/datasets/DatasetName/test3_texts.jsonl')


'''
则将tsv和jsonl文件一起序列化，转换为内存索引的LMDB数据库文件的命令如下：
python ./Chinese-CLIP/cn_clip/preprocess/build_lmdb_dataset.py --data_dir Chinese-CLIP/datasets/DatasetName --splits train,valid,test2,test3
'''

