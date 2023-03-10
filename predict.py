import torch
import config as cfg
from dataset import MyDataset
from torch.utils.data import DataLoader

testDataset = MyDataset(cfg.TEST_PATH, cfg.TEST_TRANS)
testLoader = DataLoader(testDataset, cfg.BATCH_SIZE, num_workers=cfg.N_WORKER)


