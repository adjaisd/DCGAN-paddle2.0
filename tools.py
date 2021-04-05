import os
import paddle
import paddle.nn as nn
from paddle.io import Dataset
from PIL import Image
import numpy as np
import paddle.vision.transforms as T
import cv2
import math

def tensor_to_img(data):
    if not isinstance(data,np.ndarray):
        array1=data.numpy()  #将tensor数据转为numpy数据
    else:
        array1=data
    maxValue=array1.max()
    array1=array1*255/maxValue  #normalize，将图像数据扩展到[0,255]
    mat=np.uint8(array1)  #float32-->uint8
    # print('mat_shape:',mat.shape)#mat_shape: (3, 96, 95)
    mat=mat.transpose(1,2,0)  #mat_shape: (982, 814)
    mat=cv2.cvtColor(mat,cv2.COLOR_BGR2RGB)
    return mat

def conv_initializer():
    return paddle.nn.initializer.Normal(mean=0.0, std=0.02)
def bn_initializer():
    return paddle.nn.initializer.Normal(mean=1.0, std=0.02)

def random_mini_batches(X,Y,mini_batch_size=64,seed=0):
    """
    从（X，Y）中创建一个随机的mini-batch列表

    参数：
        X - 输入数据，维度为(输入节点数量，样本的数量)
        Y - 对应的是X的标签，【1 | 0】（蓝|红），维度为(1,样本的数量)
        mini_batch_size - 每个mini-batch的样本数量

    返回：
        mini-bacthes - 一个同步列表，维度为（mini_batch_X,mini_batch_Y）

    """
    if not isinstance(X, np.ndarray):
        X=X.numpy()
    if not isinstance(Y, np.ndarray):
        Y=Y.numpy()
    np.random.seed(seed) #指定随机种子
    print(X.shape,"\tm= ",m)
    m = X.shape[0]
   
    mini_batches = []

    #第一步：打乱顺序
    perm = list(np.random.permutation(m)) #它会返回一个长度为m的随机数组，且里面的数是0到m-1
    shuffled_X = X[perm, :, :, :]   #将每一列的数据按permutation的顺序来重新排列。
    shuffled_Y = Y[perm, :]
    print("shuffled seq",shuffled_X.shape,shuffled_Y.shape)
    #第二步，分割
    num_complete_minibatches = math.floor(m / mini_batch_size) #把你的训练集分割成多少份,请注意，如果值是99.99，那么返回值是99，剩下的0.99会被舍弃
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k+1)*mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k+1)*mini_batch_size, :]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    #如果训练集的大小刚好是mini_batch_size的整数倍，那么这里已经处理完了
    #如果训练集的大小不是mini_batch_size的整数倍，那么最后肯定会剩下一些，我们要把它处理了
    if m % mini_batch_size != 0:
        #获取最后剩余的部分
        mini_batch_X = shuffled_X[mini_batch_size * num_complete_minibatches: , :, :, :]
        mini_batch_Y = shuffled_Y[mini_batch_size * num_complete_minibatches: , :]

        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches