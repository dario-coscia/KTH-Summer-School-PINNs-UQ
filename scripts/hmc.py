"""
A simple Bayesian Neural Network using HMC
Code credits: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut1_students_with_answers.html
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pyro
from pyro.infer import MCMC, NUTS, Predictive
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


# Set random seed for reproducibility
np.random.seed(111)

# Generate data
x_obs = np.hstack([np.linspace(-0.2, 0.2, 500), np.linspace(0.6, 1, 500)])
noise = 0.02 * np.random.randn(x_obs.shape[0])
y_obs = (
    x_obs
    + 0.3 * np.sin(2 * np.pi * (x_obs + noise))
    + 0.3 * np.sin(4 * np.pi * (x_obs + noise))
    + noise
)
x_true = np.linspace(-0.5, 1.5, 1000)
y_true = x_true + 0.3 * np.sin(2 * np.pi * x_true) + 0.3 * np.sin(4 * np.pi * x_true)


# The Bayesian Neural Network
class BNN(PyroModule):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=5, prior_scale=10.0):
        super().__init__()

        self.activation = nn.Tanh()
        self.layer1 = PyroModule[nn.Linear](in_dim, hid_dim)
        self.layer2 = PyroModule[nn.Linear](hid_dim, out_dim)

        # Set layer parameters as random variables
        self.layer1.weight = PyroSample(
            dist.Normal(0.0, prior_scale).expand([hid_dim, in_dim]).to_event(2)
        )
        self.layer1.bias = PyroSample(
            dist.Normal(0.0, prior_scale).expand([hid_dim]).to_event(1)
        )
        self.layer2.weight = PyroSample(
            dist.Normal(0.0, prior_scale).expand([out_dim, hid_dim]).to_event(2)
        )
        self.layer2.bias = PyroSample(
            dist.Normal(0.0, prior_scale).expand([out_dim]).to_event(1)
        )

    def forward(self, x, y=None):
        x = self.activation(self.layer1(x.reshape(-1, 1)))
        mu = self.layer2(x).squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(1, 1))
        # Sampling model
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        return mu


# Define Hamiltonian Monte Carlo (HMC) kernel
model = BNN()
hmc_kernel = NUTS(model)

# Define MCMC sampler, get 100 posterior samples
mcmc = MCMC(hmc_kernel, num_samples=100)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(x_obs).float()
y_train = torch.from_numpy(y_obs).float()

# Run MCMC
mcmc.run(x_train, y_train)

# Compute predictive distributions using posterior samples
predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
x_test = torch.linspace(-0.5, 1.5, 3000)

# Evaluate Predictive Distribution
preds = predictive(x_test)

# Take prediction samples and compute mean and std
y_pred = preds["obs"].T.detach().numpy().mean(axis=1)
y_std = preds["obs"].T.detach().numpy().std(axis=1)

# Plot Solution
fig, ax = plt.subplots(figsize=(10, 5))
xlims = [-0.5, 1.5]
ylims = [-1.5, 2.5]
plt.xlim(xlims)
plt.ylim(ylims)
plt.xlabel("X", fontsize=15)
plt.ylabel("Y", fontsize=15)
ax.plot(x_true, y_true, "b-", linewidth=3, label="true function")
ax.plot(x_obs, y_obs, "ko", markersize=4, label="observations")
ax.plot(x_obs, y_obs, "ko", markersize=3)
ax.plot(x_test, y_pred, "-", linewidth=3, color="#408765", label="predictive mean")
ax.fill_between(
    x_test, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.6, color="#86cfac", zorder=5
)
plt.legend(loc=4, fontsize=15, frameon=False)
plt.show()
