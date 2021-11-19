"""
Implementing deep convolutional GAN for MNIST
"""

import torch
from torch import random
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

class GenBlock(nn.Module):
    """
    CLass implements a single block of the generator

    Args:
        nn ([type]): [description]
    """

    def __init__(self, in_channels, out_channels, k_size=3, stride=2, is_last_block=False):
        super(GenBlock, self).__init__()

        self.is_last_block = is_last_block

        if self.is_last_block is False:
            self.cT = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=stride)
            self.bn = nn.BatchNorm2d(out_channels)
            self.act = nn.ReLU()

        # no batchnorm for final layer and tanh activation
        elif self.is_last_block is True:
            self.cT = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size, stride=stride)
            self.act = nn.Tanh()

    def forward(self, x):
        """
        Function passes input through the CNN

        Args:
            x ([type]): [description]
        """

        if self.is_last_block is True:
            x = self.act(self.cT(x))
        else:
            x = self.act(self.bn(self.cT(x)))

        return x


class Generator(nn.Module):
    """
    Implement the full generator with 3 hidden blocks and 1 output block

    Args:
        nn ([type]): [description]
    """

    def __init__(self, x):
        """
        Default method

        Args:
            x ([type]): noise vector of BxZ dims, B = batch
        """
        super(Generator, self).__init__()

        self.B = x.shape[0]
        self.C = x.shape[1]
        self.H, self.W = 1, 1
        self.hiden_dim = 64
        self.gb1 = GenBlock(in_channels=self.C, out_channels=self.hiden_dim*4)
        self.gb2 = GenBlock(in_channels=self.hiden_dim*4, out_channels=self.hiden_dim*2, k_size=4, stride=1)
        self.gb3 = GenBlock(in_channels=self.hiden_dim*2, out_channels=self.hiden_dim)
        self.gb4 = GenBlock(in_channels=self.hiden_dim, out_channels=1, k_size=4, is_last_block=True)

    def forward(self, x):
        """
        Function passes input through the NN

        Args:
            x ([type]): BxZ tensor of noise vectors
        """

        x = x.view(self.B, self.C, 1, 1)  # make into single pixel images
        x = self.gb1(x)
        x = self.gb2(x)
        x = self.gb3(x)
        x = self.gb4(x)

        return x


class DiscBlock(nn.Module):
    """
    Implements one block of the discriminator

    Args:
        nn ([type]): [description]
    """

    def __init__(self, in_ch, out_ch, k_size=4, stride=2, is_last_block=False):
        super(DiscBlock, self).__init__()

        self.is_last_block = is_last_block

        if self.is_last_block is False:
            self.c = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride)
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif self.is_last_block is True:
            self.c = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride)

    def forward(self, x):
        """
        Pass input through network

        Args:
            x ([type]): [description]
        """

        if self.is_last_block:
            x = self.c(x)
        elif self.is_last_block is False:
            x = self.act(self.bn(self.c(x)))

        return x


class Discriminator(nn.Module):
    """
    Creates a discriminator with 3 Disc blocks

    Args:
        nn ([type]): [description]
    """

    def __init__(self):
        """
        Default method, makes up the CNN body
        """
        super(Discriminator, self).__init__()

        self.hidden_dim = 16
        self.disc1 = DiscBlock(in_ch=1, out_ch=self.hidden_dim)  # first input is a 1 channel 28x28 input from the generator
        self.disc2 = DiscBlock(in_ch=self.hidden_dim, out_ch=self.hidden_dim * 2)
        self.disc3 = DiscBlock(in_ch=self.hidden_dim * 2, out_ch=1, is_last_block=True)  # finally for each input we just need a single value

    def forward(self, x):
        """
        Function passes input through the CNN

        Args:
            x ([type]): [description]
        """

        x = self.disc1(x)
        x = self.disc2(x)
        x = self.disc3(x)
        x = x.view(x.shape[0], -1)  # flatten into Bx1 since each input gets's a single label

        return x


def get_noise_vector(n_samples, n_feats):
    """
    Function returns a random normal noise vector 

    Args:
        n_samples ([type]): [description]
        n_feats ([type]): [description]
    """

    vector = torch.randn(n_samples, n_feats)
    return vector


def train_dcgan():
    """
    Run training on MNIST
    """

    BSIZE = 256
    N_FEATS = 1024

    # instantiate networks
    test_vec = get_noise_vector(BSIZE, N_FEATS)
    net_gen = Generator(test_vec).cuda()
    net_dis = Discriminator().cuda()

    criterion = nn.BCEWithLogitsLoss()

    

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset_mnist = datasets.MNIST(root='/home/sambit/data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset_mnist, batch_size=BSIZE, shuffle=True)
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    gen_opt = optim.Adam(net_gen.parameters(), lr=lr, betas=(beta1, beta2))
    dis_opt = optim.Adam(net_dis.parameters(), lr=lr, betas=(beta1, beta2))

    n_epochs = 1500
    for epoch in range(n_epochs):
        for data, labels in dataloader:
            data = data.cuda()  # send actual images to GPU
            
            #=======================  update discriminator =====================================#
            dis_opt.zero_grad()
            fake_noise = get_noise_vector(BSIZE, N_FEATS).cuda()  # get some noise data for the generator
            fake_gens = net_gen(fake_noise)  # get generator output with the noise vector, Bx1x28x28
            fake_preds_disc = net_dis(fake_gens.detach())  # get output of disc for this fake input, don't add to computation graph
            fake_preds_disc_loss = criterion(fake_preds_disc, torch.zeros_like(fake_preds_disc))  # since we know these predictions are for fake input, gt are all 0

            real_preds_disc = net_dis(data)  # now get disc predictions for the real MNIST data
            real_preds_disc_loss = criterion(real_preds_disc, torch.ones_like(real_preds_disc))  # since we know these preds are for real images, gt are all 1s
            disc_loss = (fake_preds_disc_loss + real_preds_disc_loss) / 2  # take average of both losses

            disc_loss.backward(retain_graph=True)
            dis_opt.step()

            #================= update generator ======================#
            gen_opt.zero_grad()
            fake_noise_gen = get_noise_vector(BSIZE, N_FEATS).cuda()
            fake_gens_new = net_gen(fake_noise_gen)
            preds_dis = net_dis(fake_gens_new)
            gen_loss = criterion(preds_dis, torch.ones_like(preds_dis))  # we want the generator to produce outputs that lead to all 1 from disc
            gen_loss.backward()
            gen_opt.step()

            print("epoch -> {0}, gen_loss -> {1}, dis_loss -> {2}".format(epoch, gen_loss, disc_loss))


            # save some predictions every 25 epochs
            if epoch % 1 == 0:
                sample0 = fake_gens_new[0].cpu().detach().numpy().transpose(1, 2, 0)
                plt.imsave("sample0.png", sample0[: ,:, 0])

                sample10 = fake_gens_new[10].cpu().detach().numpy().transpose(1, 2, 0)
                plt.imsave("sample10.png", sample10[: ,:, 0])

                sample20 = fake_gens_new[20].cpu().detach().numpy().transpose(1, 2, 0)
                plt.imsave("sample20.png", sample20[: ,:, 0])

                sample30 = fake_gens_new[30].cpu().detach().numpy().transpose(1, 2, 0)
                plt.imsave("sample30.png", sample30[: ,:, 0])
                pass





if __name__ == '__main__':

    test_vec = get_noise_vector(128, 64)
    print(test_vec.shape)

    gen_net = Generator(test_vec)
    outs = gen_net(test_vec)

    test_vec = torch.randn(128, 1, 28, 28)
    dis_net = Discriminator()
    outs = dis_net(test_vec)

    train_dcgan()


        