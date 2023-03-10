import torch
from torch.utils.data import DataLoader
import config as cfg
from dataset import MyDataset
from model import Generator, Discriminator
from loss import Loss
import itertools
from tqdm import tqdm

def main():
    print("Device: ", cfg.DEVIVE)

    trainDataset = MyDataset(cfg.TRAIN_PATH, cfg.TRAIN_TRANS)
    valDataset = MyDataset(cfg.VAL_PATH, cfg.TEST_TRANS)
    print("Train Images: ", trainDataset.__len__())
    print("Validation Images: ", valDataset.__len__())

    trainLoader = DataLoader(trainDataset, cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.N_WORKER)
    valLoader = DataLoader(valDataset, cfg.BATCH_SIZE, num_workers=cfg.N_WORKER)

    loss = Loss(cfg.LAMBDA_)

    netG_A2B = Generator().cuda()
    netG_B2A = Generator().cuda()
    netD_A = Discriminator().cuda()
    netD_B = Discriminator().cuda()

    G_params = list(netG_A2B.parameters()) + list(netG_B2A.parameters())
    optimizer_G = torch.optim.Adam(G_params, lr=cfg.LR, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=cfg.LR, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=cfg.LR, betas=(0.5, 0.999))

    for epoch in range(cfg.EPOCHS):
        # training
        netD_A.train()
        netD_B.train()
        netG_A2B.train()
        netG_B2A.train()

        pbar = tqdm(
            trainLoader,
            desc = "Epoch %d"%epoch,
            postfix = {'Dis_Loss': 1, 'Gen_Loss': 1},
            ncols=100
        )
        for input in pbar:

            real_A = input['A'].to(cfg.DEVIVE)
            real_B = input['B'].to(cfg.DEVIVE)

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

            # update progress bar
            dis_tot_loss = (dis_A_loss+dis_B_loss) * 0.5
            gen_tot_loss = gen_tot_loss / cfg.LAMBDA_ / 3.2
            pbar.set_postfix({
                'Dis_Loss': round(dis_tot_loss.item(), 3),
                'Gen_Loss': round(gen_tot_loss.item(), 3)
            })


if __name__=='__main__':
    main()
