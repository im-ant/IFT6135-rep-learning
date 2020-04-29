"""
Template for Question 3.
@author: Samuel Lavoie
"""

import argparse
import os

import numpy as np
import torch
from torch import optim
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

import q2_solution
from q3_sampler import svhn_sampler
from q3_model import Critic, Generator


def load_model(args, device):
    # ==
    # Initialize model
    model = Generator(z_dim=args.z_dim).to(device)

    # ==
    # Initialize state dict
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    return model


def q3_1_gen_images(args, model, device):
    # ==
    # Sample some noise
    n_batch = 64

    nsampler = torch.distributions.normal.Normal(
        loc=torch.zeros((n_batch, args.z_dim)),
        scale=torch.ones((n_batch, args.z_dim))
    )
    z_noise = nsampler.sample().to(device)  # (1, 100)

    # ==
    # Sample image
    imgs_batch = model(z_noise)

    # ==
    # Format and output
    img_grid = utils.make_grid(imgs_batch, nrow=8,
                               normalize=True)

    img_name = f'Q3-1_samples.png'
    img_path = os.path.join(args.img_out_dir, img_name)

    print(f'Saving image to: {img_path}')
    utils.save_image(img_grid, img_path)


def q3_2(args, model, device):
    # ==
    # Sample some noise
    nsampler = torch.distributions.normal.Normal(
        loc=torch.zeros((1, args.z_dim)),
        scale=torch.ones((1, args.z_dim))
    )
    z_noise = nsampler.sample().to(device)  # (1, 100)

    # ==
    # Save the images
    imgs_list = []

    # ==
    # Perturb noise
    eps = 3.0

    for dim_idx in range(args.z_dim):
        cur_z = z_noise.clone().detach()
        cur_z[0, dim_idx] += eps

        # Generate
        cur_img = model(cur_z)

        # Save the image
        imgs_list.append(cur_img)

    # ==
    # Format and output
    img_tensor = torch.cat(imgs_list, dim=0)
    img_grid = utils.make_grid(img_tensor, nrow=10,
                               normalize=True)

    img_name = f'Q3-2_perturb.png'
    img_path = os.path.join(args.img_out_dir, img_name)

    print(f'Saving image to: {img_path}')
    utils.save_image(img_grid, img_path)


def q3_3_a(args, model, device):
    # ==
    # Sample two points
    nsampler = torch.distributions.normal.Normal(
        loc=torch.zeros((1, args.z_dim)),
        scale=torch.ones((1, args.z_dim))
    )
    z_noise_a = nsampler.sample().to(device)  # (1, 100)
    z_noise_b = nsampler.sample().to(device)  # (1, 100)

    # ==
    # Interpolate between the two points

    imgs_list = []

    for alpha in np.arange(0, 1.01, 0.1):
        # Interpolate
        z1 = z_noise_a.clone().detach()
        z2 = z_noise_b.clone().detach()
        cur_z = (alpha * z1) + ((1 - alpha) * z2)

        # Generate
        cur_img = model(cur_z)

        imgs_list.append(cur_img)

    # ==
    # Format and save
    img_tensor = torch.cat(imgs_list, dim=0)
    img_grid = utils.make_grid(img_tensor, nrow=6,
                               normalize=True)

    img_name = f'Q3-3a.png'
    img_path = os.path.join(args.img_out_dir, img_name)

    print(f'Saving image to: {img_path}')
    utils.save_image(img_grid, img_path)


def q3_3_b(args):
    # ==
    # Initialize data loader
    train_loader, valid_loader, test_loader = svhn_sampler(args.data_root,
                                                           train_batch_size=2,
                                                           test_batch_size=2)

    # ==
    # Get two data points
    img_a = None
    img_b = None

    for batch_idx, train_sample in enumerate(train_loader):
        # Get real data
        real_X, real_y = train_sample

        # Divide into two
        img_a = (real_X[0, :, :, :]).unsqueeze(0)  # (1, 3, 32, 32)
        img_b = (real_X[1, :, :, :]).unsqueeze(0)

        break

    # ==
    # Interpolate between the two images

    imgs_list = []

    for alpha in np.arange(0, 1.01, 0.1):
        # Interpolate
        x1 = img_a.clone().detach()
        x2 = img_b.clone().detach()
        cur_x = (alpha * x1) + ((1 - alpha) * x2)

        imgs_list.append(cur_x)

    # ==
    # Format and save
    img_tensor = torch.cat(imgs_list, dim=0)
    img_grid = utils.make_grid(img_tensor, nrow=6,
                               normalize=True)

    img_name = f'Q3-3b.png'
    img_path = os.path.join(args.img_out_dir, img_name)

    print(f'Saving image to: {img_path}')
    utils.save_image(img_grid, img_path)


if __name__ == '__main__':
    # ==
    # (Anthony) Parsing arguments and logger
    parser = argparse.ArgumentParser(description='Training GAN')

    parser.add_argument('--model_path', type=str,
                        default='./generator_state_dict.pkl',
                        help='path to generator state dictionary')
    parser.add_argument('--img_out_dir', type=str, default='./',
                        help='image out directory path')
    parser.add_argument('--data_root', type=str, default='./',
                        help='where to put the dataset')

    parser.add_argument('--z_dim', type=int, default=100,
                        help='default latent dimension size')

    args = parser.parse_args()
    print(args)

    # ==
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # ==
    # Load model
    model = load_model(args, device)
    model.eval()
    print(f'Model loaded from: {args.model_path}')

    # ==
    # Generate things

    # q3_1_gen_images(args, model, device)  # Generate samples
    # q3_2(args, model, device)  # Latent space perturbation
    # q3_3_a(args, model, device)  # Latent space interpolation
    # q3_3_b(args)  # Image space interpolation

    """
    Note on running this:
    python q3_2-3_sols.py --model_path /network/tmp1/chenant/ant/class/ift-6135_DL/04-29_exp1/generator_state_dict.pkl --data_root $SLURM_TMPDIR --img_out_dir out_q3/q3-2-3/
    """
