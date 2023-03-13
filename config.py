import torch
import torchvision.transforms as T

IMG_SIZE = 128
DEVIVE = 'cuda' if torch.cuda.is_available() else 'cpu'

# data config
DATA_PATH = './dataset/'
TRAIN_PATH = DATA_PATH + 'train/'
VAL_PATH = DATA_PATH + 'val/'
TRAIN_TRANS = [
    T.Resize(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
]
TEST_TRANS = [
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
]

# train config
BATCH_SIZE = 1
N_WORKER = 0
EPOCHS = 100
LR = 0.0002

# predict config
TEST_PATH = ''
G_A2B_PATH = ''
G_B2A_PATH = ''

# loss lambda
LAMBDA_ = 10

# Result
WEIGHTS_PATH = 'output/weights/'
