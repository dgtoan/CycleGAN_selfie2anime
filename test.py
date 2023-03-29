import os
import glob
import torch
import config as cfg
from model import Generator
from tqdm import tqdm
from utils import pred_img

save_path = os.path.join('output/test/', cfg.MODE)
os.makedirs(save_path, exist_ok=True)

image_files = glob.glob(os.path.join(cfg.DATA_TEST_PATH, '*.*'))
transform = cfg.TEST_TRANS

model = Generator().to(cfg.DEVICE)
model.load_state_dict(torch.load(cfg.WEIGHT_PATH))

with torch.no_grad():
    model.eval()
    for img_path in tqdm(image_files):
        pred_img(img_path, model, transform=transform, save_path=save_path)
