import torch
import torch.nn as nn

class Loss():
    def __init__(self, _lambda = 10):
        self._lambda = _lambda
    
    def get_cyc_loss(self, real_data, cyc_data):
        cyc_loss = nn.L1Loss()(real_data, cyc_data)
        cyc_loss = cyc_loss * self._lambda

        return cyc_loss
    
    def get_ident_loss(self, real_data, ident_data):
        ident_loss = nn.L1Loss()(real_data, ident_data)
        ident_loss = ident_loss * (self._lambda * 0.5)

        return ident_loss
    
    def get_gen_loss(self, pred_fake_data):
        gen_taget = torch.ones_like(pred_fake_data)
        gen_loss = nn.MSELoss()(pred_fake_data, gen_taget)

        return gen_loss
    
    def get_dis_loss(self, pred_fake_data, pred_real_data):
        real_target = torch.ones_like(pred_real_data)
        fake_target = torch.zeros_like(pred_fake_data)

        loss_real_data = nn.MSELoss()(real_target, pred_real_data)
        loss_fake_data = nn.MSELoss()(fake_target, pred_fake_data)

        dis_tot_loss = (loss_fake_data + loss_real_data) * 0.5

        return dis_tot_loss
