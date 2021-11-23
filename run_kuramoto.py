#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : run_kuramoto.py
# description     : Demonstrates the link between crossfrequency coupling & IBS
# author          : Guillaume Dumas
# date            : 2021-11-09
# version         : 1
# usage           : python run_kuramoto.py
# notes           : require kuramoto.py (version by D. Laszuk)
# python_version  : 3.7-3.9
# ==============================================================================

import numpy as np
from scipy.signal import hilbert
import pylab as plt
from kuramoto import Kuramoto
import scipy.stats as st
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

plt.ion()

def simu(coupling=0.1, modulation=0.1, noise=0.1, plot=False):
    low_freq_sd = 1
    low_freq_mean = 10

    high_freq_mean = 40

    # Defining time array
    t0, t1, dt = 0, 40, 0.01
    T = np.arange(t0, t1, dt)
    
    # Y0, W, K are initial phase, intrinsic freq and
    # coupling K matrix respectively
    Y0 = np.random.rand(2)*2*np.pi
    W = np.array(np.random.randn(2) * low_freq_sd + low_freq_mean)

    W12 = coupling
    W21 = coupling
    K1 = np.array([[0, W12],
                [W12, 0]])
    K2 = np.array([[0, W21],
                [W21, 0]])

    K = np.dstack((K1, K2)).T
    
    # Passing parameters as a dictionary
    init_params = {'W':W, 'K':K, 'Y0':Y0, 'noise': 'uniform'}
    
    # Running Kuramoto model
    kuramoto = Kuramoto(init_params)
    odePhi = kuramoto.solve(T)
    
    # Computing phase dynamics
    phaseDynamics = np.diff(odePhi)/dt

    low_fb = np.sin(odePhi)
    high_fb = np.vstack((np.sin(T * 2 * np.pi * high_freq_mean) * ((1-modulation) + (modulation * low_fb[0])) + np.random.randn(4000)*noise,
                         np.sin(T * 2 * np.pi * high_freq_mean) * ((1-modulation) + (modulation * low_fb[1])) + np.random.randn(4000)*noise))

    # Plotting response
    if plot:
        nOsc = len(W)
        for osc in range(nOsc):
            plt.subplot(nOsc, 1, 1+osc)
            plt.plot(T, low_fb[osc], alpha=0.3)
            plt.plot(T, high_fb[osc], alpha=0.3)  # phaseDynamics[osc])
            plt.ylabel("$\dot\phi_{%i}$" %(osc+1))
            plt.xlim([30, 40])
        plt.show()

    high_phase = np.angle(hilbert(high_fb))
    return np.abs(np.mean(np.exp(1j * (high_phase[1] - high_phase[0]))))


def cohen_d(x, y):
    """Calculate Cohen D effect size."""
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    poolsd = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 +
                      (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return ((np.mean(x) - np.mean(y)) / poolsd)

import matplotlib.patches as mpatches
red= mpatches.Patch(color='red', label='High Coupling')
blue=  mpatches.Patch(color='blue', label='Low Coupling')

print(simu(coupling=1., modulation=0.1, noise=0.1))


for modulation in np.linspace(0,1,5):
    low_c = []
    high_c = []
    for sim in tqdm(range(100)):
        low_c.append(simu(coupling=0.01, modulation=modulation, noise=0.5))
        high_c.append(simu(coupling=0.99, modulation=modulation, noise=0.5))
    print(modulation, st.kruskal(np.array(high_c), np.array(low_c)), cohen_d(np.array(high_c), np.array(low_c)))
    plt.figure()
    plt.subplot(2,1,1); sns.violinplot(np.array(low_c), color='b')
    plt.legend(handles=[red, blue], bbox_to_anchor =(0.75, 1.25), ncol = 2)
    plt.xlim([0, 1])
    plt.xlabel('PLV')
    plt.subplot(2,1,2); sns.violinplot(np.array(high_c), color='r')
    plt.xlim([0, 1])
    plt.xlabel('PLV')
    plt.show()

# KruskalResult(statistic=555.9697458368455, pvalue=6.329078100701732e-123)


# X-axis being the inter-brain coupling in theta, Y-axis being the intra-brain cross-frequency coupling between theta and gamma, color would be the gamma inter-brain coupling

# Parameters
n_coupling = 5
n_modulation = 9
n_sims = 10
noise = 1.

# Init values ranges & grid
couplings = np.linspace(0, 1, n_coupling)
modulations = np.linspace(0, 1, n_modulation)
coupling_grid, modulation_grid = np.meshgrid(couplings, modulations, sparse=False, indexing='xy')
plv_grid = np.zeros(coupling_grid.shape) * np.nan

# Run simulations
for i_coupling in range(n_coupling):
    for i_modulation in range(n_modulation):
        # treat xv[j,i], yv[j,i]
        coupling = coupling_grid[i_modulation, i_coupling]
        modulation = modulation_grid[i_modulation, i_coupling]
        sims = [simu(coupling=coupling, modulation=modulation, noise=noise) for sim in range(n_sims)]
        plv_grid[i_modulation, i_coupling] = np.mean(np.array(sims))

# Plot results
ax = plt.subplot()
plt.imshow(plv_grid, interpolation='nearest', vmin=0, vmax=1)
ax.set_xticks(range(n_coupling))
ax.set_yticks(range(n_modulation))
ax.set_xticklabels(couplings)
ax.set_yticklabels(modulations)
plt.xlabel("Coupling in the theta band")
plt.ylabel("Modulation of gamma by theta")
plt.gca().invert_yaxis()
plt.colorbar(label='PLV')
plt.show()
