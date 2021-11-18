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
# Initial_setting
workers = 1
batch_size=64
nz = 100
nch_g = 64
nch_d = 64
n_epoch = 10000
lr = 0.001
beta1 = 0.5
outf = '../../result/EXP1_random_ads/result_lsgan'

display_interval = 100
save_fake_image_interval = 1500
plt.rcParams['figure.figsize'] = 10, 6


try:
	os.makedirs(outf, exist_ok=True)
except OSError as error:
	print(error)
	pass
 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
 
 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
	def __init__(self, nz=100, nch_g=64, nch=3):
		super(Generator, self).__init__()
		self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.ConvTranspose2d(nz, nch_g * 8, 4, 1, 0), 	
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
 
	def forward(self, z):
		for layer in self.layers.values(): 
			z = layer(z)
		return z
 
 
class Discriminator(nn.Module):
	def __init__(self, nch=3, nch_d=64):
		super(Discriminator, self).__init__()
		self.layers = nn.ModuleDict({
        	'layer0': nn.Sequential(
            	nn.Conv2d(nch, nch_d, 4, 2, 1), 
            	nn.BatchNorm2d(nch_d),
            	nn.LeakyReLU(negative_slope=0.2)	
        	),  # (3, 64, 64) -> (64, 32, 32)
        	'layer1': nn.Sequential(
            	nn.Conv2d(nch_d, nch_d * 2, 4, 2, 1),
            	nn.BatchNorm2d(nch_d*2),
            	nn.LeakyReLU(negative_slope=0.2)
        	),  # (64, 32, 32) -> (128, 16, 16)
        	'layer2': nn.Sequential(
            	nn.Conv2d(nch_d * 2, nch_d * 4, 4, 2, 1),
            	nn.BatchNorm2d(nch_d*4),
            	nn.LeakyReLU(negative_slope=0.2)
        	),  # (128, 16, 16) -> (256, 8, 8)
        	'layer3': nn.Sequential(
            	nn.Conv2d(nch_d * 4, nch_d * 8, 4, 2, 1),
            	nn.BatchNorm2d(nch_d*8),
            	nn.LeakyReLU(negative_slope=0.2)
        	),  # (256, 8, 8) -> (512, 4, 4)
        	'layer4':nn.Sequential( nn.Conv2d(nch_d * 8, 1, 4, 1, 0)
        	# (512, 4, 4) -> (1, 1, 1)
        	#勾配消失を防ぐためにSigmoidは使わない
        	)
    	})
 
	def forward(self, x):
		for layer in self.layers.values(): 
			x = layer(x)
			x = x.squeeze()
		return x
 
	
	

 
def main():
	#パスには入っている画像データセットの一個上の改装を
	#例:もし  your_home/Face_Datasets/Japanese/000.jpgのような階層になっていれば
	#root = your_home/Face_Datasetsとする
	dataset = dset.ImageFolder(root='../../dara/ads/image',
                          	transform=transforms.Compose([
                          	transforms.RandomResizedCrop(64, scale=(1.0, 1.0), ratio=(1., 1.)),
                          	transforms.RandomHorizontalFlip(),
                          	transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                          	transforms.ToTensor(),
                          	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                      	])) 
  
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                     	shuffle=True, num_workers=int(workers))
 
	#GPUがあるならGPUデバイスを作動
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	#Generatorを定義
	netG = Generator(nz=nz, nch_g=nch_g).to(device)
	#ネットワークパラメ-タをランダムに初期化
	netG.apply(weights_init)	
  
	#Discriminatorを定義
	netD = Discriminator(nch_d=nch_d).to(device)
	
	#ネットワークパラメ-タをランダムに初期化
	netD.apply(weights_init)
	
	#損失関数を二乗誤差に設定
	criterion = nn.MSELoss()  
 
	#それそれのパラメータ更新用のoptimizerを定義
	optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5) 
	optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=1e-5) 
 
	Loss_D_list, Loss_G_list = [], []
	
	fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)  # save_fake_image用ノイズ（固定）
	for epoch in range(n_epoch):
		for itr, data in enumerate(dataloader):
			real_image = data[0].to(device)   # 本物画像
			sample_size = real_image.size(0)  #画像枚数
			noise = torch.randn(sample_size, nz, 1, 1, device=device)   # 入力ベクトル生成（正規分布ノイズ）   	
			real_target = torch.full((sample_size,), random.uniform(1, 1), device=device)   # 本物ラベル
			fake_target = torch.full((sample_size,), random.uniform(0, 0), device=device)   # 偽物ラベル
        	
       	
        	#Generator→Discriminatorの順にパラメータ更新
        	
        	#---------  Update Generator   ----------
			netG.zero_grad()
			fake_image = netG(noise) # Generatorから得られた偽画像
			output = netD(fake_image)   # 更新した Discriminatorで、偽物画像を判定
			errG = criterion(output,real_target)   # 偽物画像の判定結果と本物ラベルとの二乗誤差
			errG.backward(retain_graph = True) 	# 誤差逆伝播
			D_G_z2 = output.mean().item()  # outputの平均 D_G_z2 を計算（後でログ出力に使用)
			optimizerG.step()  # Generatorのパラメータ更新
        	
        	#---------  Update Discriminaator   ----------
			netD.zero_grad() # 勾配の初期化
			fake_image = netG(noise) # Generatorから得られた偽画像
			output = netD(real_image)   # Discriminatorが行った、本物画像の判定結果
			errD_real = criterion(output,real_target)  # 本物画像の判定結果と本物ラベルとの二乗誤差
			D_x = output.mean().item()  # outputの平均 D_x を計算（後でログ出力に使用）
			
			output = netD(fake_image.detach())  # Discriminatorが行った、偽物画像の判定結果
			
			errD_fake = criterion(output,fake_target)  # 偽物画像の判定結果と偽物画像との二乗誤差
			D_G_z1 = output.mean().item()  # outputの平均 D_G_z1 を計算（後でログ出力に使用）
			
			errD =  errD_real + errD_fake	# Discriminator 全体の損失
			errD.backward(retain_graph = True)	# 誤差逆伝播
			optimizerD.step()   # Discriminatoeのパラメーター更新
        	
           	
        	#定期的に損失を表示
			if itr % 5 == 0:
				print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(x): {:.3f} D(G(z)): {:.3f}/{:.3f}'
					.format(epoch + 1, n_epoch,
							itr + 1, len(dataloader),
							errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
				
				Loss_D_list.append(errD.item())
				Loss_G_list.append(errG.item())
                          	
        	#定期的に画像を保存
			if (itr + 1) % 50 == 0:
				fake_image = netG(fixed_noise) 
				vutils.save_image(fake_image.detach(), '../../result/EXP1_random_ads/image/{:03d}random_{:03d}.png'.format(itr,epoch + 1),
                              	normalize=True, nrow=8)
 	
    	
    	
 
	# plot graph
	plt.figure()	
	plt.plot(range(len(Loss_D_list)), Loss_D_list, color='blue', linestyle='-', label='Loss_D')
	plt.plot(range(len(Loss_G_list)), Loss_G_list, color='red', linestyle='-', label='Loss_G')
	plt.legend()
	plt.xlabel('iter (*100)')
	plt.ylabel('loss')
	plt.title('Loss_D and Loss_G')
	plt.grid()
	plt.savefig('Loss_graph.png')	
 
if __name__ == '__main__':
	main()