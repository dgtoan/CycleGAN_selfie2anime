import os
import torch
import torchvision.transforms as T

IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data config
DATA_PATH = 'dataset/'
DATA_PATH=''
TRAIN_PATH = os.path.join(DATA_PATH, 'test/')
VAL_PATH = os.path.join(DATA_PATH, 'test/')
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
BATCH_SIZE = 1
N_WORKER = 0
EPOCHS = 100
LR = 0.0002

# Predict config
MODE = 'B2A'
# A2B: anime -> selfie
# B2A: selfie -> anime
TEST_PATH = 'test/B/'
RESULT_PATH = 'output/pred_imgs/'
G_A2B_PATH = 'output/weights/best_netG_A2B.pt'
G_B2A_PATH = 'output/weights/best_netG_B2A.pt'

# Cycle loss lambda
LAMBDA_ = 5

# Result
WEIGHTS_PATH = 'output/weights/'
