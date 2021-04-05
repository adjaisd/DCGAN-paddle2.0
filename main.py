import os
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.optimizer as optim
import paddle.vision.transforms as T 
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
from tools import tensor_to_img
from Dataset import DataGenerater
from unzip import unzip_file

from config import Config
import warnings
import math
import random

opt = Config()
if opt.img_size==64:
    from model64 import Generator,Discriminator
elif opt.img_size==96:
    from model96 import Generator,Discriminator
elif opt.img_size==256:
    from model256 import Generator,Discriminator
if not os.path.exists(opt.imgs_path):
    print("开始解压")
    unzip_file('data/data_test/test_faces.zip', './imgs')
    print("解压完成")

if not os.path.exists(os.getcwd() + f'/models/'):
    os.mkdir( os.getcwd() + f'/models/' )
    os.mkdir( opt.checkpoints_path )
    
warnings.filterwarnings('ignore')

paddle.disable_static()

use_gpu = paddle.is_compiled_with_cuda()
place = paddle.fluid.CUDAPlace(0) if use_gpu else paddle.fluid.CPUPlace()


if __name__=="__main__":
    batch_size=opt.batch_size
    lr=opt.lr
    z_dim = opt.z_dim
    beta1,beta2=opt.beta1,opt.beta2
    losses =[[],[]]
    real_label = paddle.full( (opt.batch_size,1,1,1), 1., dtype='float32')
    fake_label = paddle.full( (opt.batch_size,1,1,1), 0., dtype='float32')
    X = 20  #窗口大小
    #一行子窗口数量
    num=math.sqrt(batch_size)
    x=round(num) if math.fabs( math.floor(num)**2-batch_size )<1e-6 else math.floor(num)+1 

    print("start training: ")
    print("---------------------------------")
    print("num = ",num)
    
    with paddle.fluid.dygraph.guard(place):
        #损失函数
        loss = nn.BCELoss()

        netD = Discriminator(channels_img=3, features_d=10)
        netG = Generator(z_dim=z_dim, channels_img=3, features_g=10)
        optimizerD = optim.Adam(parameters=netD.parameters(), learning_rate=lr, beta1=beta1, beta2=beta2)
        optimizerG = optim.Adam(parameters=netG.parameters(), learning_rate=lr, beta1=beta1, beta2=beta2)

        if not os.path.exists( opt.checkpoints_path ):
            os.mkdir( opt.checkpoints_path )
        if not os.path.exists( opt.output_path ):
            os.mkdir( opt.output_path)
        
        last = opt.img_size
        order_name = 9
        model_path = opt.checkpoints_path+ f"model_{last}_{order_name}/"
        
        print("model path:", model_path)

        if os.path.exists(model_path):
            print("model exists")

            netD_dict, optD_dict = paddle.load(model_path+"netD.pdparams" ), \
                                                paddle.load(model_path+"adamD.pdopt"   )
            netD.set_state_dict( netD_dict )
            optimizerD.set_state_dict( optD_dict )
            print(" Model D suc")

            netG_dict, optG_dict = paddle.load(model_path+"netG.pdparams" ), \
                                                paddle.load(model_path+"adamG.pdopt"   )
            netG.set_state_dict( netG_dict )
            optimizerG.set_state_dict( optG_dict )
            print(" Model G suc")

        plt.ion()

        train_dataset = DataGenerater(opt=opt)
        train_loader  = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        print("X, x = ",X,x)
        count=0
        print("all imgs len:", len(train_dataset))
        for pass_id in range(opt.epoch):
            print(f"epotch {pass_id}: ", end=" " )
            for batch_id, (data, labels) in enumerate( tqdm(train_loader) ):
                #训练判别器 
             
                if batch_id % opt.d_every==0:
                    # print("train dis:")
                    optimizerD.clear_grad()
                    output = netD(data)
                    errD_real = loss(output, real_label)
                    errD_real.backward()
                    optimizerD.step()
                    optimizerD.clear_grad()

                    noise = paddle.randn([batch_size, z_dim, 1, 1],'float32')
                    fake = netG(noise)

                    output = netD(fake.detach())
                    errD_fake = loss(output, fake_label)
                    errD_fake.backward()
                    optimizerD.step()
                    optimizerD.clear_grad()

                    errD = errD_real + errD_fake
                    
                    losses[0].append(errD.numpy()[0])

                if batch_id % opt.g_every==0:
                    ###训练生成器
                    # print("train gen:")
                    optimizerG.clear_grad()
                    noise = paddle.randn([batch_size, z_dim, 1, 1] , 'float32')
                    fake = netG(noise)
                    
                    output = netD(fake)
                    errG = loss(output, real_label)
                    errG.backward()
                    optimizerG.step()
                    optimizerG.clear_grad()
                    
                    losses[1].append(errG.numpy()[0])
                if batch_id % 50 == 0:
                    # 每轮的生成结果
                    generated_image = netG(noise).numpy()
                    imgs=np.split(generated_image, generated_image.shape[0], 0)

                    plt.figure(figsize=(16, 4))
                    for i, ele in enumerate(imgs):
                        if i==4:
                            break
                        temp_img=ele.squeeze(0)
                        temp_img=tensor_to_img(temp_img)
                        plt.subplot(1, 4, i+1)
                        plt.axis('off')  #去掉坐标轴
                        plt.imshow(temp_img)
                    plt.savefig(opt.output_path+f"{pass_id}_{count}.jpg")
                    count+=1
                    plt.pause(1e-10)
                
            if pass_id % 2==0:
                order =  order_name+ 1+ pass_id//2
                model_path = opt.checkpoints_path + f"model_{opt.img_size}_{order}/" 
                if not os.path.exists(model_path):
                    os.mkdir(model_path)

                netD_path, optimD_path = model_path+"netD.pdparams", model_path+"adamD.pdopt"
                netD_dict, optD_dict = netD.state_dict(), optimizerD.state_dict()
                paddle.save(netD_dict , netD_path )
                paddle.save(optD_dict , optimD_path)
                
                netG_path, optimG_path = model_path+"netG.pdparams", model_path+"adamG.pdopt"
                netG_dict, optG_dict = netG.state_dict(), optimizerG.state_dict()
                paddle.save(netG_dict , netG_path )
                paddle.save(optG_dict , optimG_path)
       
