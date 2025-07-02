import tensorflow as tf
from AngioNet_model import AngioNet
import os
from PIL import Image
import os
import imagecodecs
import numpy as np

dir = os.path.join('data', "training", "images")

# from skimage import io
# img = io.imread('21_training.tif')
# print(img.shape)  # (9, 20)
# print(type(img))  # <class 'numpy.ndarray'>
# print(type(img[0][0]))  # <class 'numpy.uint16'>
# print(img.dtype)  # uint16

# data = np.array(img)
# imagecodecs.imwrite('21_training.tif', data, compression='lzw')



## 对于李响代码第一段最后部分的可能解释
# 首先，通过 X_train[...,1] 提取训练集中的 G 通道，表示只使用图像的绿色通道作为输入。
# 接着使用 crop 函数对训练数据进行裁剪，使其宽度减小 dx 个像素，这可能是为了去除图像的黑边或者其他无用的部分。
# 然后打印出 X_train 和 Y_train 的形状信息。

# 接下来，将 X_train 数组增加一维，变成形状为 (2880,1,48,48) 的四维数组，这可能是为了适配模型输入的要求。
# 然后，将 Y_train 数组的形状变成 (2880, 2304)，保持第一维不变，其他维度全部合并，这可能是为了将原本的二维标签数据转换成一维的形式。

# 接着，将 Y_train 增加一维，变成形状为 (2880, 2304, 1) 的三维数组。然后通过计算 temp = 1 - Y_train，得到 Y_train 取反后的结果。
# 最后，将 Y_train 和 temp 沿着最后一个维度拼接在一起，变成形状为 (2880, 2304, 2) 的三维数组。
# 这可能是为了将标签数据从一维转换成二维，并且对于每个像素点都有两个取值，分别对应着图像中对应位置是否属于目标类别的概率。


model = AngioNet(L1=0, L2=0, DL_weights=None)
