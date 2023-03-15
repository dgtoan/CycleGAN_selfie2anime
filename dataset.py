import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import os, glob
from PIL import Image
import config as cfg
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, data_path, transforms):
        super().__init__()
        self.transform = transforms
        self.files_A = glob.glob(data_path+'A/*.*')
        self.files_B = glob.glob(data_path+'B/*.*')
    
    def __len__(self):
        return min(len(self.files_A), len(self.files_B))
    
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index]))
        item_B = self.transform(Image.open(self.files_B[index]))

        return {'A': item_A, 'B': item_B}
    
    def plot_img(self, index):
        img_A = Image.open(self.files_A[index])
        img_B = Image.open(self.files_B[index])
        
        plt.imshow(img_A)
        plt.show()
        plt.imshow(img_B)
        plt.show()

if __name__=='__main__':
    test_dataset = MyDataset(cfg.VAL_PATH, cfg.TEST_TRANS)
    test_dataset.plot_img(5)
    print(test_dataset.__len__())
