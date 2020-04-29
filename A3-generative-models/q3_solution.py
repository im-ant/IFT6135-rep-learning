"""
Template for Question 3.
@author: Samuel Lavoie

Training code author: Anthony G. Chen
"""

import argparse
import os

import torch
from torch import optim
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter

import q2_solution
from q3_sampler import svhn_sampler
from q3_model import Critic, Generator


def train_gan(args, logger):
    # ==
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = args.data_root

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_iter = 50000  # N training iterations (default: 50,000)
    n_critic_updates = args.n_critic_updates  # N critic updates per generator update (default: 5)
    lp_coeff = args.lp_coeff  # Lipschitz penalty coefficient (default: 10)
    train_batch_size = 64  # (default: 64)
    test_batch_size = 64  # (default: 64)
    lr = args.lr  # (default: 1e-4)
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100

    # ==
    # Initialize data and models
    train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)

    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic().to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

    # ==
    # COMPLETE TRAINING PROCEDURE

    # Sampler
    nsampler = torch.distributions.normal.Normal(
        loc=torch.zeros((train_batch_size, z_dim)),
        scale=torch.ones((train_batch_size, z_dim))
    )

    # Logging and counting
    total_steps_trained = 0
    last_critic_loss = None
    last_gen_loss = None
    prev_train_X = None
    last_fp = None
    last_fq = None

    # ==
    # Training loop
    for epoch_idx in range(args.num_epochs):
        # Logging and counting

        for batch_idx, train_sample in enumerate(train_loader):
            # Get real data
            real_X, real_y = train_sample
            real_X = real_X.to(device)

            # ==
            # Critic training

            # Generator distribution
            z_noise = nsampler.sample().to(device)
            Q_X = generator(z_noise)
            Q_X = Q_X.detach()  # ensure no gradient to generator

            # Critic evaluation
            critic.zero_grad()
            wd = q2_solution.vf_wasserstein_distance(real_X, Q_X, critic)
            lp = q2_solution.lp_reg(real_X, Q_X, critic)

            critic_loss = (-wd) + (lp_coeff * lp)

            # Backprop
            critic_loss.backward()
            optim_critic.step()

            # Log critic loss
            last_wd = wd.item()
            last_critic_loss = critic_loss.item()

            # ==
            # Generator training on every nth batch
            if (batch_idx + 1) % n_critic_updates == 0:
                # Generate
                generator.zero_grad()
                z_noise = nsampler.sample().to(device)
                Q_X = generator(z_noise)

                # Critic evaluation
                critic.zero_grad()
                f_QX = critic(Q_X)

                # Generator loss
                gen_loss = (-1.0) * torch.mean(f_QX)

                gen_loss.backward()
                optim_generator.step()

                # Log loss
                last_gen_loss = gen_loss.item()

            # ==
            # Logging and counters
            total_steps_trained += 1
            prev_train_X = real_X

            # Print
            if total_steps_trained % args.print_freq == 0:
                print(f'Epoch {epoch_idx}, batch {batch_idx}, '
                      f'Gen loss: {last_gen_loss}, '
                      f'Dis loss: {last_critic_loss}, '
                      f'WD: {last_wd}')

            # Write log to Tensorboard
            if total_steps_trained % args.log_freq == 0:
                if logger is not None:
                    logger.add_scalar('Train_Loss/Wass_dist', wd.item(),
                                      total_steps_trained)
                    logger.add_scalar('Train_Loss/Lips_penalty', lp.item(),
                                      total_steps_trained)
                    logger.add_scalar('Train_Loss/Crit_loss', last_critic_loss,
                                      total_steps_trained)
                    logger.add_scalar('Train_Loss/Gen_loss', last_gen_loss,
                                      total_steps_trained)

                    # Also save model
                    mod_path = os.path.join(args.log_dir, 'generator_state_dict.pkl')
                    torch.save(generator.state_dict(), mod_path)

        # ==
        # Per epoch evaluation
        if True:
            # ==
            # Evaluate WD on the test set
            total_wd = 0.0
            num_batches = 0
            with torch.no_grad():
                for eval_sample in test_loader:
                    test_X, test_y = eval_sample
                    test_X = test_X.to(device)

                    # Generate
                    z_noise = nsampler.sample().to(device)
                    Q_X = generator(z_noise)

                    # Evaluate
                    wd = q2_solution.vf_wasserstein_distance(test_X, Q_X,
                                                             critic)

                    total_wd += wd.item()
                    num_batches += 1
            #
            print(f'Epoch {epoch_idx}, Eval WD: {total_wd / num_batches}')

            # Logger
            if logger is not None:
                logger.add_scalar('Eval_loss/Wass_dist', (total_wd / num_batches),
                                  total_steps_trained)

            # ==
            # Generate images

            # To Tensorboard
            if logger is not None:
                if (epoch_idx + 1) % args.img_log_freq == 0:
                    with torch.no_grad():
                        # Generate
                        z_noise = nsampler.sample().to(device)
                        cur_Q = generator(z_noise)

                        # Organize
                        gen_img_grid = utils.make_grid(cur_Q[0:8], nrow=4,
                                                       normalize=True)
                        logger.add_image('Generated_Image', gen_img_grid,
                                         total_steps_trained)

                        # Also save some training images
                        train_img_grid = utils.make_grid(prev_train_X[0:4],
                                                         nrow=4,
                                                         normalize=True)
                        logger.add_image('Training_Image', train_img_grid,
                                         total_steps_trained)

            # Directly to a directory
            if args.img_out_dir is not None:
                with torch.no_grad():
                    # Generate
                    z_noise = nsampler.sample().to(device)
                    cur_Q = generator(z_noise)

                    # Organize
                    img_grid = utils.make_grid(cur_Q, normalize=True)
                    img_name = f'epoch-{epoch_idx}_steps-{total_steps_trained}_Q.png'
                    img_path = os.path.join(args.img_out_dir, img_name)

                    print(f'Saving image to: {img_path}')
                    utils.save_image(img_grid, img_path)


if __name__ == '__main__':
    # ==
    # (Anthony) Parsing arguments and logger
    parser = argparse.ArgumentParser(description='Training GAN')

    parser.add_argument('--data_root', type=str, default='./',
                        help='path to directory for data download (default: ./)')
    parser.add_argument('--img_out_dir', type=str, default=None,
                        help='path to output image directory every '
                             'epoch (default: None)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='path to log directory (default: None)')

    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of training steps per print-out')
    parser.add_argument('--n_critic_updates', type=int, default=5,
                        help='num critic update per generator update '
                             '(default: 5)')
    parser.add_argument('--lp_coeff', type=float, default=10.0,
                        help='Lipschitz penalty coefficient (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 1e-4)')

    parser.add_argument('--print_freq', type=int, default=200,
                        help='number of training steps per print-out')
    parser.add_argument('--log_freq', type=int, default=200,
                        help='number of training steps per write to log')
    parser.add_argument('--img_log_freq', type=int, default=5,
                        help='number of epochs per image save')

    args = parser.parse_args()
    print(args)

    # Tensorboard
    logger = None
    if args.log_dir is not None:
        logger = SummaryWriter(args.log_dir)

    # ==
    # Training
    train_gan(args, logger)

    # ==
    # COMPLETE QUALITATIVE EVALUATION

    # NOTE: done in separate script
