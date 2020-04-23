"""
Template for Question 1 of hwk3.
@author: Shawn Tan and Jae Hyun Lim
"""
import math
import numpy as np
import torch


def log_likelihood_bernoulli(mu, target):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Bernoulli random variables p(x=1).
    :param target: (FloatTensor) - shape: (batch_size x input_size) - Target samples (binary values).
    :return: (FloatTensor) - shape: (batch_size,) - log-likelihood of target samples on the Bernoulli random variables.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    target = target.view(batch_size, -1)

    # ==
    # Bernoulli log likelihood
    ele_log_prob = ((target * mu.log())
                    + ((1-target) * (1-mu).log()))  # (batch, input)
    sample_log_prob = torch.sum(ele_log_prob, dim=1)  # (batch, )

    # log_likelihood_bernoulli
    return sample_log_prob


def log_likelihood_normal(mu, logvar, z):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu: (FloatTensor) - shape: (batch_size x input_size) - The mean of Normal distributions.
    :param logvar: (FloatTensor) - shape: (batch_size x input_size) - The log variance of Normal distributions.
    :param z: (FloatTensor) - shape: (batch_size x input_size) - Target samples.
    :return: (FloatTensor) - shape: (batch_size,) - log probability of the sames on the given Normal distributions.
    """
    # init
    batch_size = mu.size(0)
    mu = mu.view(batch_size, -1)
    logvar = logvar.view(batch_size, -1)
    z = z.view(batch_size, -1)

    # ==
    # Gaussian log likelihood
    ele_rel = logvar + ((z-mu).pow(2) / logvar.exp())  # (batch, input)
    ele_inv_ll = np.log(2*np.pi) + ele_rel  # add the log2pi constant
    batch_ll = (-1.0/2.0) * torch.sum(ele_inv_ll, dim=1)  # (batch, )

    # NOTE maybe TODO: recheck my derivation of the log likelihood is correct
    # passed unit test but just to be sure

    # log normal
    return batch_ll


def log_mean_exp(y):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param y: (FloatTensor) - shape: (batch_size x sample_size) - Values to be evaluated for log_mean_exp. For example log proababilies
    :return: (FloatTensor) - shape: (batch_size,) - Output for log_mean_exp.
    """
    # init
    batch_size = y.size(0)
    sample_size = y.size(1)

    # ==
    # log mean exp
    ymax = y.max(dim=1, keepdim=True)[0]  # (batch, 1)
    exp_y = (y-ymax).exp()  # (batch, sample)
    sum_exp = torch.sum(exp_y, dim=1)  # (batch, )
    lme = ((1.0/sample_size) * sum_exp).log() + ymax.view(-1)

    # NOTE maybe TODO: check mean is outside of sum (passed test but be sure)

    # log_mean_exp
    return lme


def kl_gaussian_gaussian_analytic(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    mu_q = mu_q.view(batch_size, -1)
    logvar_q = logvar_q.view(batch_size, -1)
    mu_p = mu_p.view(batch_size, -1)
    logvar_p = logvar_p.view(batch_size, -1)

    # ==
    # Gaussian analytical KLdiv
    ele_kl = (logvar_p - logvar_q - 1.0
              + (logvar_q.exp() / logvar_p.exp())
              + ((mu_q - mu_p).pow(2) / logvar_p.exp()))  # (batch, input)
    kld = (1.0/2.0) * torch.sum(ele_kl, dim=1)  # (batch, )

    # kld
    return kld


def kl_gaussian_gaussian_mc(mu_q, logvar_q, mu_p, logvar_p, num_samples=1):
    """ 
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** note. ***

    :param mu_q: (FloatTensor) - shape: (batch_size x input_size) - The mean of first distributions (Normal distributions).
    :param logvar_q: (FloatTensor) - shape: (batch_size x input_size) - The log variance of first distributions (Normal distributions).
    :param mu_p: (FloatTensor) - shape: (batch_size x input_size) - The mean of second distributions (Normal distributions).
    :param logvar_p: (FloatTensor) - shape: (batch_size x input_size) - The log variance of second distributions (Normal distributions).
    :param num_samples: (int) - shape: () - The number of sample for Monte Carlo estimate for KL-divergence
    :return: (FloatTensor) - shape: (batch_size,) - kl-divergence of KL(q||p).
    """
    # init
    batch_size = mu_q.size(0)
    input_size = np.prod(mu_q.size()[1:])
    mu_q = mu_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_q = logvar_q.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    mu_p = mu_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)
    logvar_p = logvar_p.view(batch_size, -1).unsqueeze(1).expand(batch_size, num_samples, input_size)

    # ==
    # Monte carlo kld

    # Take sample from Q
    stdev_q = (((1.0/2.0) * logvar_q).exp())
    normal = torch.distributions.Normal(mu_q.float(), stdev_q.float())
    z = normal.sample()

    # Compute the relative density under Q and P
    q_den = logvar_q + ((z-mu_q).pow(2) / logvar_q.exp())
    p_den = logvar_p + ((z-mu_p).pow(2) / logvar_p.exp())
    rel_den = q_den - p_den

    # Compute KL
    batch_sample_sum = torch.sum(rel_den, dim=2)  # (batch, num_samples)
    batch_sum = torch.sum(batch_sample_sum, dim=1)  # (batch, )
    kld = (-1.0/2.0) * (1.0/num_samples) * batch_sum

    # TODO NOTE maybe fix: there may be catastrophic cancellation happening
    # sometimes get negative values 

    # kld
    return kld
