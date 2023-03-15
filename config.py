import os
import torch
import torchvision.transforms as T

IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data config
DATA_PATH = 'dataset/'
TRAIN_PATH = os.path.join(DATA_PATH, 'train/')
VAL_PATH = os.path.join(DATA_PATH, 'val/')
TRAIN_TRANS = T.Compose([
    T.Resize(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
TEST_TRANS = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Train config
BATCH_SIZE = 8
N_WORKER = 2
EPOCHS = 50
LR = 0.0002

# Predict config
MODE = 'B2A'
# A2B: anime -> selfie
# B2A: selfie -> anime
TEST_PATH = 'test/B/'
RESULT_PATH = 'output/pred_imgs/'
G_A2B_PATH = 'output/weights/best_netG_A2B.pt'
G_B2A_PATH = 'output/weights/best_netG_B2A.pt'

# Loss lambda
LAMBDA_ = 10

# Result
WEIGHTS_PATH = 'output/weights/'
