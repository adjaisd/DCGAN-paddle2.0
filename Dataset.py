import os
import paddle
from paddle.io import Dataset
from PIL import Image
import paddle.vision.transforms as T
import cv2
from config import Config

opt=Config()
class DataGenerater(Dataset):
    def __init__(self,opt=opt):
        super(DataGenerater, self).__init__()
        self.dir = opt.imgs_path
        
        self.datalist = os.listdir(self.dir) if opt.test==False else os.listdir(self.dir)[:100]
        self.batch_size=opt.batch_size

        img=Image.open(self.dir+self.datalist[0])
        self.image_size = img.size
        img.close()
    
        self.transform=T.Compose([
            T.Resize(opt.img_size),
            T.CenterCrop(opt.img_size),
            T.ToTensor(),
        ])
        self.num_path_dict={}
    
    # 每次迭代时返回数据和对应的标签
    def __getitem__(self, idx):
        path=self.dir + self.datalist[idx]
        img=cv2.imread(path)
        if self.transform:
            img=self.transform(img)
        self.num_path_dict[idx]=path
        return (img, idx)

    def get_img_path(self, idx):
        return self.num_path_dict[idx]


    # 返回整个数据集的总数
    def __len__(self):
        return len(self.datalist)
