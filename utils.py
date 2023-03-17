import os
import torch
import config as cfg
from PIL import Image
from torchvision.utils import (
    make_grid,
    save_image
)

def save_weights(netD_A = None, netD_B = None, netG_A2B = None, netG_B2A = None, type_='last'):
    if netD_A != None:
        torch.save(netD_A.state_dict(), os.path.join(cfg.WEIGHTS_PATH, type_+'_netD_A.pt'))
    if netD_B != None:
        torch.save(netD_B.state_dict(), os.path.join(cfg.WEIGHTS_PATH, type_+'_netD_B.pt'))
    if netG_A2B != None:
        torch.save(netG_A2B.state_dict(), os.path.join(cfg.WEIGHTS_PATH, type_+'_netG_A2B.pt'))
    if netG_B2A != None:
        torch.save(netG_B2A.state_dict(), os.path.join(cfg.WEIGHTS_PATH, type_+'_netG_B2A.pt'))

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
