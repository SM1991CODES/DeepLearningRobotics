import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


def plot_subplots(fig_list=None, labels=None, direction='h'):
    """
    Function plots the 1 or 3-channel figures using subplots
    :param labels: labels for the subplots
    :param fig_list: list of 1 or 3 channel figures
    :return: None
    """

    if fig_list == None:
        return

    num_plots = len(fig_list)
    if direction == 'v':
        fig, axs = plt.subplots(num_plots)
    else:
        fig, axs = plt.subplots(1, num_plots)

    fig.suptitle('Vertically stacked range images')
    for i in range(num_plots):
        axs[i].imshow(fig_list[i])  # depth channel of the 3 channel normalized range image

        if labels is not None:
            axs[i].set_title(labels[i])
    plt.show()
    return fig


class Generator(nn.Module):
    """
    Implements the generator part of the GAN

    Args:
        nn ([type]): [description]
    """

    def __init__(self, in_feats_size, out_feats_size):
        """
        Default constructor, makes up the NN body

        Args:
            in_feats_size ([type]): size of the input noise vector
            out_feats_size: size of output features size (24x24 -> 784)
        """
        super(Generator, self).__init__()

        self.lin1 = nn.Linear(in_features=in_feats_size, out_features=256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()

        self.lin2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()

        self.lin3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu3 = nn.ReLU()

        self.lin4 = nn.Linear(1024, 784)
        self.bn4 = nn.BatchNorm1d(784)
        self.sig4 = nn.Sigmoid()

    def forward(self, x):
        """
        Function passes the input through the network

        Args:
            x ([type]): [description]
        """

        x = self.relu1(self.bn1(self.lin1(x)))
        x = self.relu2(self.bn2(self.lin2(x)))
        x = self.relu3(self.bn3(self.lin3(x)))
        x = self.sig4(self.bn4(self.lin4(x)))

        return x


class Discriminator(nn.Module):
    """
    Class implements the discriminator NN

    Args:
        nn ([type]): [description]
    """

    def __init__(self, in_feats_size):
        """
        Default constructor

        Args:
            in_feats_size ([type]): [description]
        """
        super(Discriminator, self).__init__()

        self.lin1 = nn.Linear(in_feats_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.LeakyReLU(0.02)

        self.lin2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.LeakyReLU(0.02)

        self.lin3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.act3 = nn.LeakyReLU(0.02)

        self.lin4 = nn.Linear(64, 1)
        self.act4 = nn.Sigmoid()


    def forward(self, x):
        """
        Function passes inputs through the NN

        Args:
            x ([type]): [description]
        """

        x = self.act1(self.bn1(self.lin1(x)))
        x = self.act2(self.bn2(self.lin2(x)))
        x = self.act3(self.bn3(self.lin3(x)))

        x = self.act4(self.lin4(x))

        return x


def train_GAN():
    """
    Function trains the GAN network and returns the trained model for generator
    """

    DEBUG_PLOTS = 0

    net_gen = Generator(784, 784).cuda()
    net_dis = Discriminator(784).cuda()

    # download the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_mnist = MNIST(root='/home/sambit/data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset_mnist, batch_size=4096)

    criterion = nn.BCEWithLogitsLoss()
    optim_gen = optim.Adam(net_gen.parameters())
    optim_dis = optim.Adam(net_dis.parameters())

    noise_in_dim = 784

    for epoch in range(2000):
        for index, batch in enumerate(dataloader):
            data, labels = batch
            data = data.view(-1, 784).cuda()
            data_noise = torch.randn(data.shape[0], noise_in_dim).cuda()  # noise, each element in the batch has 784 elements, this is what we decided as input dims to gen
            data_noise /= torch.max(data_noise)
            # data_noise = torch.normal(0, 0.5, size=(data.shape[0], noise_in_dim)).cuda()
            gt_fake = torch.zeros(data.shape[0], 1).cuda()  # for fake images we know the gt class is always 0
            gt_real = torch.ones(data.shape[0], 1).cuda()

            # --------------- discriminator update part -----------------------#
            optim_dis.zero_grad()
            gen_preds_fake = net_gen(data_noise)  # generate some fake images
            dis_preds_fake = net_dis(gen_preds_fake.detach())  # get discrim. output for these fake images, don't add that to grad computation for generator
            dis_fake_loss = criterion(dis_preds_fake, gt_fake)  # loss for the fake inputs

            dis_preds_real = net_dis(data)  # get discriminator preds for actual images
            dis_real_loss = criterion(dis_preds_real, gt_real)  # loss for the real data class predictions, for real data gt is always 1
            loss_dis = (dis_fake_loss + dis_real_loss) / 2  # average of the 2 losses
            loss_dis.backward(retain_graph=True)
            optim_dis.step()
            #------------------------------------------------------------------------------------#

            # ------------------- generator part ---------------------------------#
            optim_gen.zero_grad()
            data_noise_gen = torch.randn(data.shape[0], noise_in_dim).cuda()
            data_noise_gen /= torch.max(data_noise_gen)
            gt_gen = torch.ones(data.shape[0], 1).cuda()  # for generator, we always want gt to be 1 and gen's output to match gt
            gen_preds = net_gen(data_noise_gen)
            dis_preds = net_dis(gen_preds)
            gen_loss = criterion(dis_preds, gt_gen)
            gen_loss.backward()
            optim_gen.step()

            print("epoch --> {2}, dis loss -> [{0}], gen loss -> [{1}]".format(loss_dis.item(), gen_loss.item(), epoch))

            if epoch >= 1500:
                print("HALTTT")
                gen_preds_npy = gen_preds.cpu().detach().numpy()[0].reshape(28, 28)
                plt.imshow(gen_preds_npy)

            if DEBUG_PLOTS:
                img1 = data[0].numpy().transpose(1, 2, 0)
                label1 = str(labels[0].numpy().item())
            
                img2 = data[1].numpy().transpose(1, 2, 0)
                label2 = str(labels[1].numpy().item())
                plot_subplots([img1, img2], [label1, label2])






if __name__ == '__main__':
    train_GAN()

