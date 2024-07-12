from sklearn.model_selection import train_test_split
import os
from PIL import Image
import cv2
import shutil


def LoadProstateXDataset():
    data = []
    image_names = []
    image_dir = "/home/david/dataset/ProstateSeg/ProstateX/images/"
    imgs = os.listdir(image_dir)

    for item in imgs:
        # Get height, width
        # print(image_dir+"/"+str(i)+".tif")
        img = Image.open(image_dir + item)
        h = img.size[0]
        w = img.size[1]
        # index = item.rfind('.png')
        # name = item[:index]
        d = {"image": item, "height": h, "width": w}
        data.append(d)
        image_names.append(item)  # 将图像文件名添加到新的列表中

    return data, image_names  # 返回两个列表


data, image_names = LoadProstateXDataset()  # 接收两个返回值

random_state = 2021
split_ratio_train_test = 0.2
train_data, test_data = train_test_split(image_names, test_size=split_ratio_train_test, random_state=random_state)

data_dir = "/home/david/dataset/ProstateSeg/ProstateX/images/"
mask_dir = "/home/david/dataset/ProstateSeg/ProstateX/masks"

train_path = "/home/david/dataset/ProstateSeg/ProstateX/train/images"
os.makedirs(train_path, exist_ok=True)
test_path = "/home/david/dataset/ProstateSeg/ProstateX/test/images"
os.makedirs(test_path, exist_ok=True)

train_mask_p = "/home/david/dataset/ProstateSeg/ProstateX/train/masks"
os.makedirs(train_mask_p, exist_ok=True)
test_mask_p = "/home/david/dataset/ProstateSeg/ProstateX/test/masks"
os.makedirs(test_mask_p, exist_ok=True)

# 循环复制数据和掩码
for dataset, new_path, mask_path in [(train_data, train_path, train_mask_p), (test_data, test_path, test_mask_p)]:
    for image_name in dataset:
        # 复制数据
        shutil.copy(os.path.join(data_dir, image_name), os.path.join(new_path, image_name))
        # 复制掩码
        shutil.copy(os.path.join(mask_dir, image_name), os.path.join(mask_path, image_name))

print(1)
