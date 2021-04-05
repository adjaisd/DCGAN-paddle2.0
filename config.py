import os
class Config:
    img_size=256
    
    lr = 0.0002

    z_dim = 100    # 噪声维度

    g_every = 4 # 每 5个batch训练一次生成器
    d_every = 2 # 每 3个batch训练一次判别器
    
    test = False

    epoch = 100
    batch_size = 24
    
    beta1=0.5
    beta2=0.999
    imgs_path = os.getcwd() + "/imgs/test_faces/"

    # output='/root/paddlejob/workspace/output'
    output = os.getcwd()
    output_path = output + '/output/'
    checkpoints_path= output + f'/models/models_{img_size}/'
