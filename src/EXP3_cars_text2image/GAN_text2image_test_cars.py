# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import copy
from collections import OrderedDict
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

sample_size = 64
nz = 100
nch_g = 64

class Generator(nn.Module):
    
    def __init__(self, nz=100, nch_g=64, nch=3):
        super(Generator, self).__init__()
        self.embed_dim = 768
        self.projected_embed_dim = 128
        self.latent_dim = nz + self.projected_embed_dim

        self.projection = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
            nn.BatchNorm1d(num_features=self.projected_embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.ConvTranspose2d(self.latent_dim, nch_g * 8, 4, 1, 0),     
                nn.BatchNorm2d(nch_g * 8),                      
                nn.ReLU()                                    
            ),  # (100, 1, 1) -> (512, 4, 4)
            'layer1': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 8, nch_g * 4, 4, 2, 1),
                nn.BatchNorm2d(nch_g * 4),
                nn.ReLU()     
            ),  # (512, 4, 4) -> (256, 8, 8)
            'layer2': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 4, nch_g * 2, 4, 2, 1),
                nn.BatchNorm2d(nch_g * 2),
                nn.ReLU()  
            ),  # (256, 8, 8) -> (128, 16, 16)
            'layer3': nn.Sequential(
                nn.ConvTranspose2d(nch_g * 2, nch_g, 4, 2, 1),
                nn.BatchNorm2d(nch_g),
                nn.ReLU()
            ),  # (128, 16, 16) -> (64, 32, 32)
            'layer4': nn.Sequential(
                nn.ConvTranspose2d(nch_g, nch, 4, 2, 1),
                nn.Tanh()
            )   # (64, 32, 32) -> (3, 64, 64)
        })
 
    def forward(self, embed_vector, z):
        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        z = torch.cat([projected_embed, z], 1)
        for layer in self.layers.values(): 
            z = layer(z)
        return z


def main():
    cnt = 0
    model = SentenceTransformer("../../data/cars/transformers/")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    noise = torch.randn(sample_size, nz, 1, 1, device=device)   # ????????????????????????????????????????????????)
    #Generator?????????
    netG = Generator(nz=nz, nch_g=nch_g).to(device)
    #???????????????????????????-???????????????
    netG.load_state_dict(torch.load('../../result/EXP3_cars_text2image/model/899gen_006.pth', map_location=torch.device(device)))
    with torch.no_grad():
        while True:
            #????????????????????????3?????????
            text = input("3 colors>")
            tmp = text.split(" ")
            embed = model.encode(tmp[0])
            embed = torch.tensor(embed)
            embed = torch.reshape(embed, (1, 768)).to(device)
            for i in range(sample_size - 1):
                if i < 23:                
                    sentence_embedding = model.encode(tmp[0])
                elif i > 38:
                    sentence_embedding = model.encode(tmp[2])
                else:
                    sentence_embedding = model.encode(tmp[1])
                sentence_embedding = torch.tensor(sentence_embedding)
                sentence_embedding = torch.reshape(sentence_embedding, (1, 768)).to(device)
                embed = torch.cat((embed, sentence_embedding), 0)
            netG.zero_grad()
            image = netG(embed, noise)
            vutils.save_image(image.detach(), '../../result/EXP3_cars_text2image/output/output_{:03d}.png'.format(cnt), normalize=True, nrow=8)
            cnt += 1
if __name__ == '__main__':
    main()
