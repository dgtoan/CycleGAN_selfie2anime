import os
import shutil
import torch
import config as cfg
from PIL import Image
from torchvision.utils import (
    make_grid,
    save_image
)

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

if __name__=='__main__':
    pass
