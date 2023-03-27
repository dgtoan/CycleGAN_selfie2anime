import os
import torch
import torchvision.transforms as T

IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data config
DATA_PATH = 'dataset/'
TRAIN_TRANS = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
TEST_TRANS = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Train config
BATCH_SIZE = 8
N_WORKER = 4
EPOCHS = 200
EPOCH_DECAY = 100
GEN_LR = 0.0002
DIS_LR = 0.0002

# Predict config
MODE = 'B2A'
    # A2B: anime -> selfie
    # B2A: selfie -> anime
DATA_TEST_PATH = 'test/B/'
WEIGHT_PATH = 'output/run/weights/best_netG_A2B.pt'

# Cycle loss lambda
LAMBDA_ = 10
