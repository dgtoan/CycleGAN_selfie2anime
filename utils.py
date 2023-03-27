import os
import shutil
import torch
import config as cfg
import numpy as np
from PIL import Image
from torchvision.utils import (
    make_grid,
    save_image
)

class ImageBuffer():
    def __init__(self, buffer_sz = 50):
        self.buffer_sz = buffer_sz
        self.buffer = []
        self.n_images = 0
        
    def push_pop(self, images):
        buffer_return = []
        
        for image in images:
            image = torch.unsqueeze(image, 0)
            
            if self.n_images < self.buffer_sz:
                self.buffer.append(image)
                buffer_return.append(image)
                self.n_images += 1
            else:
                if np.random.uniform(0, 1) > 0.5:
                    rand_idx = np.random.randint(0, self.buffer_sz)
                    temp_img = self.buffer[rand_idx].clone()
                    self.buffer[rand_idx] = image
                    buffer_return.append(temp_img)
                else:
                    buffer_return.append(image)
        
        return torch.cat(buffer_return, 0)

def pred_img(img_path, model, transform=cfg.TEST_TRANS, save_img=False):
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0) # CHW -> NCHW
    img = img.to(cfg.DEVICE)

    pred_img = model(img)
    pred_img = make_grid(pred_img, normalize=True)

    if save_img:
        pred_img_path = cfg.RESULT_PATH + os.path.split(img_path)[-1]
        save_image(pred_img, pred_img_path)

    return pred_img

def make_grid_images(*imgs):
    imgs = torch.cat((imgs))
    imgs = make_grid(imgs, normalize=True, nrow=4)
    return imgs

def make_folders(root_dir='output/run', *paths):
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    
    for path in paths:
        os.makedirs(path, exist_ok=True)

def lr_lambda(epoch):
    assert cfg.EPOCH_DECAY < cfg.EPOCHS
    fraction = (epoch - cfg.EPOCH_DECAY) / (cfg.EPOCHS - cfg.EPOCH_DECAY)
    return 1.0 - max(0, fraction)

def weights_init_normal(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant(model.bias.data, 0.0)


if __name__=='__main__':
    pass
