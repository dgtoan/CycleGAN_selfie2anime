import os
import torch
from torch.utils.data import DataLoader
from config import *
from dataset import MyDataset
from model import Generator, Discriminator
from loss import Loss
from tqdm import tqdm
from utils import (
    save_weights,
)

def main():
    os.makedirs(WEIGHTS_PATH, exist_ok=True)
    print("Weights save in:", WEIGHTS_PATH)
    print("Device: ", DEVICE)

    trainDataset = MyDataset(TRAIN_PATH, TRAIN_TRANS)
    valDataset = MyDataset(VAL_PATH, TEST_TRANS)
    trainLoader = DataLoader(trainDataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKER)
    valLoader = DataLoader(valDataset, BATCH_SIZE, num_workers=N_WORKER)
    trainData_len = trainDataset.__len__()
    valData_len = valDataset.__len__()
    print("Train Images: ", trainData_len)
    print("Validation Images: ", valData_len)
    print()

    netG_A2B = Generator().to(DEVICE)
    netG_B2A = Generator().to(DEVICE)
    netD_A = Discriminator().to(DEVICE)
    netD_B = Discriminator().to(DEVICE)

    G_params = list(netG_A2B.parameters()) + list(netG_B2A.parameters())
    optimizer_G = torch.optim.Adam(G_params, lr=LR, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=LR, betas=(0.5, 0.999))

    loss = Loss(LAMBDA_)
    losses_train = []
    losses_val = []
    min_loss = 1.0

    for epoch in range(EPOCHS):
        print('EPOCH', epoch)
        running_train_loss = [0.0]*4
        running_val_loss = [0.0]*4

        # TRAINING
        print('Training:')
        netD_A.train()
        netD_B.train()
        netG_A2B.train()
        netG_B2A.train()

        for input in tqdm(trainLoader, ncols=80):

            real_A = input['A'].to(DEVICE)
            real_B = input['B'].to(DEVICE)

            # Generator
            optimizer_G.zero_grad()

            fake_A = netG_B2A(real_B)
            pred_fake_A = netD_A(fake_A)
            genB2A_loss = loss.get_gen_loss(pred_fake_A)

            fake_B = netG_A2B(real_A)
            pred_fake_B = netD_B(fake_B)
            genA2B_loss = loss.get_gen_loss(pred_fake_B)

            cyc_A = netG_B2A(fake_B)
            cyc_A_loss = loss.get_cyc_loss(real_A, cyc_A)
            cyc_B = netG_A2B(fake_A)
            cyc_B_loss = loss.get_cyc_loss(real_B, cyc_B)

            same_A = netG_B2A(real_A)
            ident_A_loss = loss.get_ident_loss(real_A, same_A)
            same_B = netG_A2B(real_B)
            ident_B_loss = loss.get_ident_loss(real_B, same_B)

            gen_tot_loss = genA2B_loss + genB2A_loss + cyc_A_loss + cyc_B_loss + ident_A_loss + ident_B_loss

            gen_tot_loss.backward()
            optimizer_G.step()

            # Discriminator A
            optimizer_D_A.zero_grad()

            pred_real_A = netD_A(real_A)
            pred_fake_A = netD_A(fake_A.detach())
            dis_A_loss = loss.get_dis_loss(pred_fake_A, pred_real_A)

            dis_A_loss.backward()
            optimizer_D_A.step()

            # Discriminator B
            optimizer_D_B.zero_grad()

            pred_real_B = netD_B(real_B)
            pred_fake_B = netD_B(fake_B.detach())
            dis_B_loss = loss.get_dis_loss(pred_fake_B, pred_real_B)

            dis_B_loss.backward()
            optimizer_D_B.step()

            # Log training loss
            running_train_loss[0] += genA2B_loss.item()
            running_train_loss[1] += dis_B_loss.item()
            running_train_loss[2] += genB2A_loss.item()
            running_train_loss[3] += dis_A_loss.item()
            running_train_loss *= real_A.size(0)

        running_train_loss[0] = round(running_train_loss[0]/trainData_len, 3)
        running_train_loss[1] = round(running_train_loss[1]/trainData_len, 3)
        running_train_loss[2] = round(running_train_loss[2]/trainData_len, 3)
        running_train_loss[3] = round(running_train_loss[3]/trainData_len, 3)
        print('GenA2B Loss: {}, DisB Loss: {}\nGenB2A Loss: {}, DisA Loss: {}'.format(*running_train_loss))
        losses_train.append(running_train_loss)

        # VALIDATING
        print('Validating:')

        netD_A.eval()
        netD_B.eval()
        netG_A2B.eval()
        netG_B2A.eval()

        with torch.no_grad():
            for input in tqdm(valLoader, ncols=80):
                real_A = input['A'].to(DEVICE)
                real_B = input['B'].to(DEVICE)

                # Generator loss
                fake_A = netG_B2A(real_B)
                pred_fake_A = netD_A(fake_A)
                genB2A_loss = loss.get_gen_loss(pred_fake_A)
                fake_B = netG_A2B(real_A)
                pred_fake_B = netD_B(fake_B)
                genA2B_loss = loss.get_gen_loss(pred_fake_B)

                cyc_A = netG_B2A(fake_B)
                cyc_A_loss = loss.get_cyc_loss(real_A, cyc_A)
                cyc_B = netG_A2B(fake_A)
                cyc_B_loss = loss.get_cyc_loss(real_B, cyc_B)

                same_A = netG_B2A(real_A)
                ident_A_loss = loss.get_ident_loss(real_A, same_A)
                same_B = netG_A2B(real_B)
                ident_B_loss = loss.get_ident_loss(real_B, same_B)

                gen_tot_loss = genA2B_loss + genB2A_loss + cyc_A_loss + cyc_B_loss + ident_A_loss + ident_B_loss

                # Discrimiator loss
                pred_real_A = netD_A(real_A)
                pred_fake_A = netD_A(fake_A.detach())
                dis_A_loss = loss.get_dis_loss(pred_fake_A, pred_real_A)

                pred_real_B = netD_B(real_B)
                pred_fake_B = netD_B(fake_B.detach())
                dis_B_loss = loss.get_dis_loss(pred_fake_B, pred_real_B)

                # Log training loss
                running_val_loss[0] += genA2B_loss.item()
                running_val_loss[1] += dis_B_loss.item()
                running_val_loss[2] += genB2A_loss.item()
                running_val_loss[3] += dis_A_loss.item()
                running_val_loss *= real_A.size(0)

            running_val_loss[0] = round(running_val_loss[0]/valData_len, 3)
            running_val_loss[1] = round(running_val_loss[1]/valData_len, 3)
            running_val_loss[2] = round(running_val_loss[2]/valData_len, 3)
            running_val_loss[3] = round(running_val_loss[3]/valData_len, 3)
            print('GenA2B Loss: {}, DisB Loss: {}\nGenB2A Loss: {}, DisA Loss: {}'.format(*running_val_loss))
            losses_train.append(running_val_loss)

        # Save model
        save_weights(netD_A, netD_B, netG_A2B, netG_B2A, type_='last')

        if (running_val_loss[0]+running_val_loss[1])/2 < min_loss and abs(running_val_loss[0]-running_val_loss[1]) < 0.05:
            min_loss = (running_val_loss[0]+running_val_loss[1])/2

            save_weights(netD_A, netD_B, netG_A2B, netG_B2A, type_='best')
            print('Best weights saved!')
        print()

if __name__=='__main__':
    main()
