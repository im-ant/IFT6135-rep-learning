"""
Template for Question 2 of hwk3.
@author: Samuel Lavoie
"""
import torch
import q2_sampler
import q2_model


def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """

    # ==
    # Lipschitz penalty

    # Compute sample to use
    unif = torch.distributions.uniform.Uniform(0.0, 1.0)
    u = unif.sample()
    x_hat = (u * x) + ((1-u) * y)
    x_hat = x_hat.clone().detach().requires_grad_(True)

    # Send sample through function
    f_x = critic(x_hat)

    # Compute gradient of sample
    ones = torch.ones(f_x.size()).to(f_x.device)  # placeholder
    grads = torch.autograd.grad(outputs=f_x, inputs=x_hat,
                                grad_outputs=ones,
                                create_graph=True, retain_graph=True,
                                only_inputs=True)[0]  # (batch, *, ...)
    grads = grads.view(grads.size(0), -1)  # (batch, features)

    # Compute nrm and penalty
    dx_norm = torch.norm(grads, p=2, dim=1,
                         keepdim=False)  # (batch, )

    zeros = torch.zeros(dx_norm.size()).to(x_hat.device)
    grad_penalty = (torch.max(zeros, (dx_norm - 1.0))).pow(2)  # (batch, )
    
    # Compute Lipschitz penalty
    lp = torch.mean(grad_penalty)

    return lp


def vf_wasserstein_distance(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """

    # ==
    # Wasserstein distance

    f_P = critic(x)  # real samples, (batch, )
    f_Q = critic(y)  # generated samples, (batch, )

    # Estimate empirical Wasserstein distance
    wd = torch.mean(f_P) - torch.mean(f_Q)

    return wd


def vf_squared_hellinger(x, y, critic):
    """
    Complete me. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Nowazin et al: https://arxiv.org/pdf/1606.00709.pdf
    In other word, x are samples from the distribution P and y are samples from the distribution Q. Please note that the Critic is unbounded. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Squared Hellinger.
    :return: (FloatTensor) - shape: (1,) - Estimate of the Squared Hellinger
    """

    # ==
    # Squared Hellinger objective

    # Compute V(x)'s
    v_P = critic(x)  # (batch, )
    v_Q = critic(y)  # (batch, )

    # Compute g(v)
    gv_P = 1.0 - (-v_P).exp()  # (batch, )
    gv_Q = 1.0 - (-v_Q).exp()  # (batch, )

    # Compute -f(g)
    neg_f = - ((gv_Q) / (1.0 - gv_Q))  # (batch, )

    # Empirical average Squared Hellinger loss
    sqh = torch.mean((gv_P + neg_f))

    return sqh


if __name__ == '__main__':
    # ==
    # Example of usage of the code provided for answering Q2.5 as well as recommended hyper parameters.
    model = q2_model.Critic(2)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler1 = iter(q2_sampler.distribution1(0, 512))
    theta = 0
    sampler2 = iter(q2_sampler.distribution1(theta, 512))
    lambda_reg_lp = 50  # Recommended hyper parameters for the lipschitz regularizer.
