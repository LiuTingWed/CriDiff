import os
from PIL import Image
import numpy as np

# 设置数据集的路径
dataset_path = '/home/david/dataset/ProstateSeg/ProstateX_move_black'  # 将此路径替换为你的数据集路径
masks_path = os.path.join(dataset_path, 'test/masks')
images_path = os.path.join(dataset_path, 'test/images')
# real_masks_path = os.path.join(dataset_path, 'real_masks/train')

# 获取所有掩膜文件
mask_files = os.listdir(masks_path)
threshold = 0.01  # 将此阈值设置为你认为适合的值

# 遍历每一个掩膜文件
for mask_file in mask_files:
    # 打开掩膜文件
    mask = Image.open(os.path.join(masks_path, mask_file))
    # 将掩膜转换为numpy数组
    mask_array = np.array(mask)
    # 检查掩膜是否全黑
    if np.mean(mask_array) < threshold:
        # 如果掩膜全黑，则删除对应的掩膜和图像
        corresponding_image_file = mask_file  # 假设图像文件和掩膜文件有相同的文件名
        os.remove(os.path.join(masks_path, mask_file))
        os.remove(os.path.join(images_path, corresponding_image_file))
        # os.remove(os.path.join(real_masks_path, corresponding_image_file))
