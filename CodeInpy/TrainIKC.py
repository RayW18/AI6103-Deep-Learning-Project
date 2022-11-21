from model import *
from Functions import *
from dataset import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from torch.autograd import Variable
import torch.nn as nn


class Train_IKC():
    def main(self):
        FixedPath = r'/content/drive/MyDrive/Colab Notebooks/DeepLearningFinalProject/Results/Result1/testIKC.pth'
        train_data = train_dataset(self.DIR_PATH, self.DATASET_DIR)
        load_path = self.DIR_PATH + self.F_CKPT_FIR_NAME
        net_p = Predictor().to('cuda')

        optimizer_p = torch.optim.Adam(net_p.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-07,
                                       weight_decay=0.0005)
        net_c = Corrector().to('cuda')

        optimizer_c = torch.optim.Adam(net_c.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-07,
                                       weight_decay=0.0005)

        net_f = SFTMD().to('cuda')
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                [k[7:]] = v
            else:
                load_net_clean[k] = v
        net_f.load_state_dict(load_net_clean, strict=True)

        checkpoint_path_predictor = self.DIR_PATH + self.P_CKPT_FIR_NAME
        checkpoint_path_corrector = self.DIR_PATH + self.F_CKPT_FIR_NAME
        checkpoint_dir = self.DIR_PATH + "CodeInpy/model/"

        epoch = 100
        psnr_psftmd_epoch = []
        psnr_ikc_epoch = []

        for epoch in range(epoch):
            print("Epoch ", epoch)
            trainloader = torch.utils.data.DataLoader(train_data, batch_size=self.Batch_Size, shuffle=True, num_workers=2)
            psnr_psftmd_batch = []
            psnr_ikc_batch = []
            count_batch = 0

            for batch_idx, inputs in enumerate(trainloader):
                O_B, O_H, O_W, O_C = inputs.size()
                count_batch += 1
                LR_img, ker_map = prepro(inputs)
                LR_img = LR_img.to('cuda')

                optimizer_p.zero_grad()
                fake_ker = net_p(LR_img.to("cuda")).to("cuda")
                # fake_ker =  torch.FloatTensor(2*np.random.randn(O_B,10)).to("cuda")
                criterion1 = nn.MSELoss().to('cuda')
                loss1 = criterion1(fake_ker, ker_map)
                loss1.backward()
                optimizer_p.step()
                est_ker_map = fake_ker.detach().float().cpu().to('cuda')
                est_ker_map = est_ker_map.to('cuda')

                SR_img = net_f(LR_img, est_ker_map).detach().float().cpu().to('cuda')

                print(f"epoch {epoch} batch_idx {batch_idx} Predictor Loss {loss1}")

                psnr_iter = []
                for steps in range(10):
                    optimizer_c.zero_grad()

                    delta_h = net_c(SR_img, est_ker_map.to('cuda'))
                    # h0 = est_ker_map.to('cuda') + delta_h.to('cuda')
                    criterion2 = nn.MSELoss().to('cuda')
                    loss2 = criterion2(delta_h.to('cuda'), ker_map.to('cuda'))
                    loss2.backward()
                    optimizer_c.step()
                    est_ker_map = delta_h.detach().float().cpu()

                    SR_img = net_f(LR_img, est_ker_map.to('cuda')).detach().float().cpu().to('cuda')

                    # --------- image loss for experiments ---------#
                    criterion3 = nn.L1Loss().to('cuda')
                    loss3 = criterion3(SR_img.to('cuda'), inputs.to('cuda'))
                    print(f"epoch {epoch} batch_idx {batch_idx} step {steps} Corrector Loss {loss2} Image Loss {loss3}")
                  
        # save training model for predictor and corrector
        torch.save(net_p.state_dict(), checkpoint_path_predictor)
        torch.save(net_c.state_dict(), checkpoint_path_corrector)
        # torch.save(net_p, checkpoint_path_predictor)
        # torch.save(net_c, checkpoint_path_corrector)

    def __init__(self):
        dic = read_yaml('main.yaml')
        self.DIR_PATH = dic['DIR_PATH']
        self.DATASET_DIR = dic['DATASET_DIR']
        self.F_CKPT_FIR_NAME = dic['F_CKPT_FIR_NAME']
        self.P_CKPT_FIR_NAME = dic['NEW_P_CKPT_FIR_NAME']
        self.C_CKPT_FIR_NAME = dic['NEW_C_CKPT_FIR_NAME']
        self.Batch_Size = dic['ICK_Batch_Size']