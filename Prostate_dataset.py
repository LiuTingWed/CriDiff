import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageStat
from torchvision import transforms as T
from functools import partial
from pathlib import Path
from torch import nn
import albumentations as A
import cv2


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def exists(x):
    return x is not None


class Dataset(data.Dataset):
    def __init__(
            self,
            folder,
            image_size,
            mode,
            convert_image_to,
            # flag_remove_all_,
            exts=['jpg', 'jpeg', 'png', 'tiff'],

    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.mode = mode
        # maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        self.convert_image_to = convert_image_to

        # mean_list = []
        # std_list = []
        # img_train_paths= os.path.join(folder, "images/train/")
        # for i in np.arange(0, 1521):  # Prostate has 1521 images with name form 1.png to 1521.png
        #     # Get height, width
        #     # print(image_dir+"/"+str(i)+".tif")
        #     img  = cv2.imread(str(img_train_paths + "/" + str(i + 1) + ".png"),0)
        #     img_mean = np.mean(img)
        #     img_std = np.std(img)
        #     mean_list.append(img_mean)
        #     std_list.append(img_std)
        #
        # img_test_paths= os.path.join(folder, "images/test/")
        # for i in np.arange(0, 271):  # Prostate has 1521 images with name form 1.png to 1521.png
        #     # Get height, width
        #     # print(image_dir+"/"+str(i)+".tif")
        #     img  = cv2.imread(str(img_test_paths + "/" + str(i + 1) + ".png"),0)
        #     img_mean = np.mean(img)
        #     img_std = np.std(img)
        #     mean_list.append(img_mean)
        #     std_list.append(img_std)
        #
        # self.mean = np.mean(mean_list)
        # self.std = np.mean(std_list)
        self.mean = np.mean(51.93044825547968)
        self.std = np.mean(39.16262095837929)
        assert mode == 'train' or mode == 'test'
        if mode == 'train':
            img_paths = os.path.join(folder, "images/train/")
            self.img_paths = [p for ext in exts for p in Path(f'{img_paths}').glob(f'**/*.{ext}')]
            mask_paths = os.path.join(folder, "masks/train/")
            self.mask_paths = [p for ext in exts for p in Path(f'{mask_paths}').glob(f'**/*.{ext}')]
            body_paths = os.path.join(folder, "body/train/")
            self.body_paths = [p for ext in exts for p in Path(f'{body_paths}').glob(f'**/*.{ext}')]
            detail_paths = os.path.join(folder, "detail/train/")
            self.detail_paths = [p for ext in exts for p in Path(f'{detail_paths}').glob(f'**/*.{ext}')]

            self.transform = A.Compose([
                # A.Lambda(image=maybe_convert_fn),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.5, rotate_limit=90, border_mode=0, value=0, p=0.5),
                # A.RandomCrop(height=image_size[0], width=image_size[1], p=1),
                A.Resize(height=320, width=320, interpolation=cv2.INTER_NEAREST),
                A.RandomCrop(height=image_size, width=image_size),
                # A.Resize(height=256, width=256, interpolation=cv2.INTER_NEAREST),
                # A.Normalize(mean,std)
                # ToTensorV2()
            ], additional_targets={'body': 'mask', 'detail': 'mask'})
        elif mode == 'test':
            img_paths = os.path.join(folder, "images/test/")
            self.img_paths = [p for ext in exts for p in Path(f'{img_paths}').glob(f'**/*.{ext}')]
            mask_paths = os.path.join(folder, "masks/test/")
            self.mask_paths = [p for ext in exts for p in Path(f'{mask_paths}').glob(f'**/*.{ext}')]

            import re

            def sort_by_number(path):
                numbers = re.findall(r'\d+', str(path))
                return int(numbers[-1]) if numbers else 0

            self.img_paths = sorted(self.img_paths, key=sort_by_number)
            self.mask_paths = sorted(self.mask_paths, key=sort_by_number)
            self.transform = A.Compose([
                # A.Lambda(image=maybe_convert_fn),
                A.Resize(height=256, width=256),
                # ToTensorV2()
            ])
        else:
            raise ValueError

    def Normalize(self, image, mask=None, body=None, detail=None):
        # image = (image - self.mean) / self.std
        if mask is None:
            return image
        if body is None:
            return image, mask / 255
        return image, mask / 255, body/255, detail/255

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        if self.mode == 'train':
            img_paths = self.img_paths[index]
            img = Image.open(img_paths)
            img = convert_image_to_fn(self.convert_image_to, img)

            mask_paths = self.mask_paths[index]
            mask = Image.open(mask_paths)
            mask = convert_image_to_fn(self.convert_image_to, mask)

            body_paths = self.body_paths[index]
            body = Image.open(body_paths)
            body = convert_image_to_fn(self.convert_image_to, body)
            # body = convert_image_to_fn(self.convert_image_to, body).astype('int32')

            detail_paths = self.detail_paths[index]
            detail = Image.open(detail_paths)
            detail = convert_image_to_fn(self.convert_image_to, detail)
            # detail = convert_image_to_fn(self.convert_image_to, detail).astype('int32')

            # 提取文件名
            img_name = os.path.basename(img_paths)
            mask_name = os.path.basename(mask_paths)
            body_name = os.path.basename(body_paths)
            detail_name = os.path.basename(detail_paths)

            # 检查文件名是否相同
            assert img_name == mask_name == body_name == detail_name, \
                f"Files do not match: {img_name}, {mask_name}, {body_name}, {detail_name}"
            # body = np.where(np.array(body) > 1, 1, np.array(body))
            # detail = np.where(np.array(detail) >1, 1, np.array(detail))
            # transform = self.transform(image=np.array(img), mask=np.array(mask), body=body,
            #                            detail=detail)

            transform = self.transform(image=np.array(img), mask=np.array(mask), body=np.array(body),
                                       detail=np.array(detail))
            image_data = transform['image']
            mask_data = transform['mask']
            body_data = transform['body']
            detail_data = transform['detail']

            # cv2.imwrite('./image_data.jpg', image_data)
            # cv2.imwrite('./mask_data.jpg', mask_data)
            # cv2.imwrite('./body_data.jpg', body_data*255)
            # cv2.imwrite('./detail_data.jpg', detail_data*255)
            mask_data = np.where(mask_data > 64, 255, 0)
            image_data, mask_data, body_data, detail_data = self.Normalize(image_data, mask_data, body_data,
                                                                           detail_data)

            image_data = torch.from_numpy(image_data).float().unsqueeze(dim=0)
            mask_data = torch.from_numpy(mask_data).float().unsqueeze(dim=0)
            body_data = torch.from_numpy(body_data).float().unsqueeze(dim=0)
            detail_data = torch.from_numpy(detail_data).float().unsqueeze(dim=0)

            if torch.all(mask_data==0):
                class_lable = torch.tensor([0.], dtype=torch.float32)
            else:
                class_lable = torch.tensor([1.], dtype=torch.float32)

            # check for vis_numpy
            image_data2 = np.array(image_data)
            mask_data2 = np.array(mask_data)
            body_data2 = np.array(body_data)
            detail_data2 = np.array(detail_data)

            # return image_data, mask_data, body_data, detail_data, class_lable
            return image_data, mask_data, body_data, detail_data

        elif self.mode == 'test':
            img_paths = self.img_paths[index]
            img = Image.open(img_paths)
            img = convert_image_to_fn(self.convert_image_to, img)

            mask_paths = self.mask_paths[index]
            mask = Image.open(mask_paths)
            mask = convert_image_to_fn(self.convert_image_to, mask)

            # 提取文件名
            img_name = os.path.basename(img_paths)
            mask_name = os.path.basename(mask_paths)

            # 检查文件名是否相同
            assert img_name == mask_name, \
                f"Files do not match: {img_name}, {mask_name}"

            transform = self.transform(image=np.array(img), mask=np.array(mask))
            image_data = transform['image']
            mask_data = transform['mask']
            mask_data = np.where(mask_data > 64, 255, 0)
            image_data, mask_data = self.Normalize(image_data, mask_data)

            image_name = str(self.img_paths[index]).split('/')[-1].split('.')[0]

            image_data = torch.from_numpy(image_data).float().unsqueeze(dim=0)
            mask_data = torch.from_numpy(mask_data).float().unsqueeze(dim=0)

            if torch.all(mask_data==0):
                class_lable = torch.tensor([0.], dtype=torch.float32)
            else:
                class_lable = torch.tensor([1.], dtype=torch.float32)

            # check for vis_numpy
            image_data2 = np.array(image_data)
            mask_data2 = np.array(mask_data)
            # body_data2 = np.array(body_data)
            # detail_data2 = np.array(detail_data)

            # return image_data, mask_data, class_lable, image_name
            return image_data, mask_data, image_name

        else:
            raise ValueError


if __name__ == '__main__':

    train_dataset = Dataset("/home/ubuntu/data/ProstateV2", 256, 'train', convert_image_to='L')
    test_dataset = Dataset("/home/ubuntu/data/ProstateV2", 256, 'test', convert_image_to='L')

    test_que = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, drop_last=False,
        pin_memory=True, shuffle=True)

    train_qus = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, drop_last=False,
        pin_memory=True, shuffle=True)

    for i, (data) in enumerate(train_qus):
        # image_data, mask_data, name = data
        image_data, mask_data, body_data, detail_data = data
        print(i)
