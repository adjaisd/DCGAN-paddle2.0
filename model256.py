#定义了生成器和判别器的类
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from tools import conv_initializer,bn_initializer

class Discriminator(nn.Layer):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()

        # Input : N x C x 256 x 256
        self.disc=nn.Sequential(
            nn.Conv2D(                                              # 128 x 128
                channels_img, features_d, kernel_size=4, stride=2, padding=1,
                weight_attr=paddle.ParamAttr(initializer=conv_initializer())
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d   , features_d*2, 4, 2, 1),      # 64 x 64 
            self._block(features_d*2 , features_d*4, 4, 2, 1),      # 32 x 32
            self._block(features_d*4 , features_d*8, 4, 2, 1),      # 16 x 16
            self._block(features_d*8 , features_d*16, 4, 2, 1),     # 8 x 8
            self._block(features_d*16, features_d*32, 4, 2, 1),     # 4 x 4
            nn.Conv2D(  features_d*32, 1, kernel_size=4, stride=2, padding=0,# 1 x 1 
                weight_attr=paddle.ParamAttr(initializer=conv_initializer() ) 
            ),
            nn.Sigmoid(),
        )
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2D(
                in_channels, out_channels, kernel_size, stride, padding, bias_attr=False, 
                weight_attr=paddle.ParamAttr(initializer=conv_initializer() ) 
            ),
            nn.LeakyReLU(0.2),
        )
    def forward(self, input):
        return self.disc(input)

class Generator(nn.Layer):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen=nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim         , features_g*64 , 4, 1, 0),   # N x f_g x 4 x 4
            self._block(features_g*64 , features_g*32 , 4, 2, 1),   # N x f_g x 8  x 8
            self._block(features_g*32 , features_g*16 , 4, 2, 1),   # N x f_g x 16 x 16
            self._block(features_g*16 , features_g*8  , 4, 2, 1),   # N x f_g x 32 x 32
            self._block(features_g*8  , features_g*4  , 4, 2, 1),   # N x f_g x 64 x 64
            self._block(features_g*4  , features_g*2  , 4, 2, 1),   # N x f_g x 128 x 128
            nn.Conv2DTranspose(
                features_g*2, channels_img, kernel_size=4, stride=2, padding=1, bias_attr=False, 
                weight_attr=paddle.ParamAttr(initializer=conv_initializer() )
            ),
            nn.Tanh()   # [-1, 1]
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2DTranspose(
                in_channels, out_channels, kernel_size, stride, padding, bias_attr=False, 
                weight_attr=paddle.ParamAttr(initializer=conv_initializer() )
            ),
            nn.BatchNorm2D(
                out_channels, 
                weight_attr=paddle.ParamAttr(initializer=bn_initializer() ),
                momentum=0.8
            ),
            nn.ReLU(),
        )
    def forward(self, input):
        return self.gen(input)

# def test():
#     N, C, H, W= 8, 3, 256,256
#     z_dim = 100
#     X=paddle.randn( (N, C, H, W ))
#     disc = Discriminator(C, N)
#     print("1:",disc(X).shape)
#     assert disc(X).shape == [N, 1, 1 ,1]

#     z=paddle.randn( (N, z_dim, 1, 1),dtype="float32")
#     gen=Generator(z_dim, C, N)
#     print("2:",gen(z).shape)

# test()
