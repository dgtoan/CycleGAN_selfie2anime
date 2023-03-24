import os
import glob
import torch
import config as cfg
from model import Generator
from tqdm import tqdm
import torchvision.transforms as T
from utils import pred_img

os.makedirs(cfg.RESULT_PATH, exist_ok=True)
image_files = glob.glob(cfg.DATA_TEST_PATH+'/*.*')
model = Generator().to(cfg.DEVICE)
transform = T.Compose(cfg.TEST_TRANS)

if cfg.MODE == 'A2B':
    model.load_state_dict(torch.load(cfg.G_A2B_PATH))
    print("From anime to selfie image")
elif cfg.MODE == 'B2A':
    model.load_state_dict(torch.load(cfg.G_B2A_PATH))
    print("From selfie image to anime")
else:
    print("ERROR: Predict mode does not exist!")
    exit()

with torch.no_grad():
    model.eval()
    for img_path in tqdm(image_files):
        pred_img(img_path, model, save_img=True)
