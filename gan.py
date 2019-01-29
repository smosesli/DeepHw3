from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        # To extract image features you can use the EncoderCNN from the VAE
        # section or implement something new.
        # You can then use either an affine layer or another conv layer to
        # flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        kernel_size = (5, 5)
        stride = (2, 2)
        filters = (in_size[0], 64, 128, 256, 512)
        for i, (in_filters, out_filters) in enumerate(zip(filters, filters[1:]), start=1):
            modules.append(nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=stride))
            modules.append(nn.LeakyReLU(negative_slope=0.2))
            if i is not 1:
                modules.append(nn.BatchNorm2d(out_filters))
        self.cnn = nn.Sequential(*modules)
        self.affine = nn.Linear(filters[-1]*4*4, 1)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        # No need to apply sigmoid to obtain probability - we'll combine it
        # with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        bs = x.shape[0]
        x = self.cnn(x)
        y = self.affine(x.view(bs, -1))
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        # To combine image features you can use the DecoderCNN from the VAE
        # section or implement something new.
        # You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.fm_size = featuremap_size
        kernel_size = (5, 5)
        stride = (2, 2)
        self.filters = (512, 256, 128, 64, out_channels)
        modules = []
        self.affine = nn.Linear(z_dim, self.filters[0] * self.fm_size * self.fm_size)
        for i, (in_filters, out_filters) in enumerate(zip(self.filters, self.filters[1:]), start=1):
            modules.append(nn.ConvTranspose2d(in_filters, out_filters, kernel_size=kernel_size,
                                              stride=stride, padding=stride, output_padding=(1, 1)))
            if i is not len(self.filters):
                modules.append(nn.ReLU())
                modules.append(nn.BatchNorm2d(out_filters))
            else:
                modules.append(nn.Tanh())
        self.cnn = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should have
        gradients or not.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        # Generate n latent space samples and return their reconstructions.
        # Don't use a loop.
        # ====== YOUR CODE: ======
        torch.autograd.set_grad_enabled(mode=with_grad)
        z = torch.randn(n, self.z_dim, device=device)
        samples = self.forward(z)
        torch.autograd.set_grad_enabled(mode=True)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        # Don't forget to make sure the output instances have the same scale
        # as the original (real) images.
        # ====== YOUR CODE: ======
        x = self.affine(z)
        x = self.cnn(x.view(x.shape[0], self.filters[0], self.fm_size, self.fm_size))
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO: Implement the discriminator loss.
    # See torch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    r = label_noise / 2
    noisy_label_data = torch.FloatTensor(y_data.size()).uniform_(data_label - r, data_label + r).to(y_data.device)
    noisy_label_generated = torch.FloatTensor(y_generated.size()).uniform_((1 - data_label) - r, (1 - data_label) + r).to(y_generated.device)
    loss_data = F.binary_cross_entropy_with_logits(y_data, noisy_label_data)
    loss_generated = F.binary_cross_entropy_with_logits(y_generated, noisy_label_generated)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    # TODO: Implement the Generator loss.
    # Think about what you need to compare the input to, in order to
    # formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    loss = F.binary_cross_entropy_with_logits(y_generated, torch.full(y_generated.size(), data_label).to(y_generated.device))
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: torch.Tensor):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    bs = x_data.shape[0]
    x_gen = gen_model.sample(bs, with_grad=True)
    dsc_optimizer.zero_grad()
    y_data = dsc_model(x_data)
    y_gen = dsc_model(x_gen.detach())
    dsc_loss = dsc_loss_fn(y_data, y_gen)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    y_gen2 = dsc_model(x_gen)
    gen_loss = gen_loss_fn(y_gen2)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()

