import os
import itertools
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import *
from dataset import MyDataset
from model import Generator, Discriminator
from loss import Loss
from tqdm import tqdm
from utils import (
    make_grid_images,
    make_folders,
    lr_lambda,
    ImageBuffer,
    weights_init_normal,
)


def main():
    print("Device: ", DEVICE)

    trainDataset = MyDataset(os.path.join(DATA_PATH, 'train/'), TRAIN_TRANS)
    valDataset = MyDataset(os.path.join(DATA_PATH, 'val/'), TEST_TRANS)
    trainLoader = DataLoader(trainDataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKER)
    valLoader = DataLoader(valDataset, BATCH_SIZE, shuffle=True, num_workers=N_WORKER)
    trainData_len = trainDataset.__len__()
    valData_len = valDataset.__len__()
    print("Train Images: ", trainData_len)
    print("Validation Images: ", valData_len)
    
    run_dir = 'output/run/'
    weight_save_path = os.path.join(run_dir, 'weights')
    tensorboard_path = os.path.join(run_dir, 'tensorboard')
    make_folders(run_dir, weight_save_path, tensorboard_path)
    print('Output save in', run_dir)
    print()
    
    writer = SummaryWriter(tensorboard_path)

    netG_A2B = Generator().to(DEVICE)
    netG_B2A = Generator().to(DEVICE)
    netD_A = Discriminator().to(DEVICE)
    netD_B = Discriminator().to(DEVICE)
    
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # G_params = list(netG_A2B.parameters()) + list(netG_B2A.parameters())
    G_params = itertools.chain(netG_A2B.parameters(), netG_B2A.parameters())
    optimizer_G = torch.optim.Adam(G_params, lr=GEN_LR, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=DIS_LR, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=DIS_LR, betas=(0.5, 0.999))
    
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda)
    loss = Loss(LAMBDA_)
    
    fake_A_buffer = ImageBuffer()
    fake_B_buffer = ImageBuffer()

    for epoch in range(EPOCHS):
        print('EPOCH {}/{}'.format(epoch, EPOCHS))
        show_imgs = True
        running_train_loss = np.zeros(5)
        running_val_loss = np.zeros(5)

        # TRAINING
        print('Training:')
        netD_A.train()
        netD_B.train()
        netG_A2B.train()
        netG_B2A.train()

        for input in tqdm(trainLoader, ncols=60):

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

            fake_A = fake_A_buffer.push_pop(fake_A)
            pred_real_A = netD_A(real_A)
            pred_fake_A = netD_A(fake_A.detach())
            dis_A_loss = loss.get_dis_loss(pred_fake_A, pred_real_A)

            dis_A_loss.backward()
            optimizer_D_A.step()

            # Discriminator B
            optimizer_D_B.zero_grad()

            fake_B = fake_B_buffer.push_pop(fake_B)
            pred_real_B = netD_B(real_B)
            pred_fake_B = netD_B(fake_B.detach())
            dis_B_loss = loss.get_dis_loss(pred_fake_B, pred_real_B)

            dis_B_loss.backward()
            optimizer_D_B.step()

            temp_all_loss = np.array([
                gen_tot_loss.item(),
                (dis_A_loss+dis_B_loss).item(),
                (genA2B_loss+genB2A_loss).item(),
                (ident_A_loss+ident_B_loss).item(),
                (cyc_A_loss+cyc_B_loss).item()
            ])
            running_train_loss += temp_all_loss * real_A.size(0)

        running_train_loss = np.around(running_train_loss/trainData_len, 4)
        print( 'Gen Total Loss: {}\
                \nDis Total Loss: {}\
                \nGen Loss: {}, Identity Loss: {}, Cycle Loss: {}'.format(*running_train_loss))

        # VALIDATING
        print('Validating:')

        netD_A.eval()
        netD_B.eval()
        netG_A2B.eval()
        netG_B2A.eval()

        with torch.no_grad():
            for input in tqdm(valLoader, ncols=60):
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

                temp_all_loss = np.array([
                    gen_tot_loss.item(),
                    (dis_A_loss+dis_B_loss).item(),
                    (genA2B_loss+genB2A_loss).item(),
                    (ident_A_loss+ident_B_loss).item(),
                    (cyc_A_loss+cyc_B_loss).item()
                ])
                running_val_loss += temp_all_loss * real_A.size(0)
        
        running_val_loss = np.around(running_val_loss/valData_len, 4)
        print( 'Gen Total Loss: {}\
                \nDis Total Loss: {}\
                \nGen Loss: {}, Identity Loss: {}, Cycle Loss: {}\n'.format(*running_val_loss))
        
        if show_imgs:
            grid_imgs = make_grid_images(real_A[:2], real_B[:2], fake_B[:2], fake_A[:2])
            writer.add_image('images', grid_imgs, epoch)
            show_imgs = False

        writer.add_scalars(
            "Total Loss/Generator",
            {"Training": running_train_loss[0], "Validating": running_val_loss[0]},
            epoch
        )
        writer.add_scalars(
            "Total Loss/Discriminator",
            {"Training": running_train_loss[1], "Validating": running_val_loss[1]},
            epoch
        )
        writer.add_scalars(
            "Detail Gen Loss/Gen",
            {"Training": running_train_loss[2], "Validating": running_val_loss[2]},
            epoch
        )
        writer.add_scalars(
            "Detail Gen Loss/Identity",
            {"Training": running_train_loss[3], "Validating": running_val_loss[3]},
            epoch
        )
        writer.add_scalars(
            "Detail Gen Loss/Cycle",
            {"Training": running_train_loss[4], "Validating": running_val_loss[4]},
            epoch
        )
        writer.flush()
        
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save model
        torch.save(netD_A.state_dict(), os.path.join(weight_save_path, 'netD_A.pt'))
        torch.save(netD_B.state_dict(), os.path.join(weight_save_path, 'netD_B.pt'))
        torch.save(netG_A2B.state_dict(), os.path.join(weight_save_path, 'netG_A2B.pt'))
        torch.save(netG_B2A.state_dict(), os.path.join(weight_save_path, 'netG_B2A.pt'))
     
    writer.close()

if __name__=='__main__':
    main()
