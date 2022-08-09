#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : run_kuramoto.py
# description     : Demonstrates the link between crossfrequency coupling & IBS
# author          : Guillaume Dumas, Quentin Moreau
# date            : 2022-04-07
# version         : 2
# usage           : python run_kuramoto.py
# notes           : require kuramoto.py (version by D. Laszuk)
# python_version  : 3.7-3.9
# ==============================================================================


from copyreg import pickle
from matplotlib.colorbar import Colorbar
import numpy as np
from scipy.signal import hilbert
import pylab as plt
from kuramoto import Kuramoto
import scipy.stats as st
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('Qt5Agg') 


plt.ion()

def simu(coupling=0.1, modulation=0.1, noise=0.1, plot=False):
    low_freq_sd = 1
    low_freq_mean = 6

    high_freq_mean = 40

    # Defining time array
    t0, t1, dt = 0, 40, 0.01
    T = np.arange(t0, t1, dt)
    
    # Y0, W, K are initial phase, intrinsic freq and
    # coupling K matrix respectively
    Y0 = np.random.rand(2)*2*np.pi
    W = 2 * np.pi * np.array(np.random.randn(2) * low_freq_sd + low_freq_mean)

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
n_coupling = 11
n_modulation = 11
n_sims = 3
noise = 0

# Init values ranges & grid
couplings = np.linspace(0, 1, n_coupling)
modulations = np.linspace(0, 1, n_modulation)
coupling_grid, modulation_grid = np.meshgrid(couplings, modulations, sparse=False, indexing='xy')
plv_grid = np.zeros(coupling_grid.shape) * np.nan
plv_std = np.zeros(coupling_grid.shape) * np.nan

# Run simulations
for i_coupling in range(n_coupling):
    for i_modulation in range(n_modulation):
        # treat xv[j,i], yv[j,i]
        coupling = coupling_grid[i_modulation, i_coupling]
        modulation = modulation_grid[i_modulation, i_coupling]
        sims = [simu(coupling=coupling, modulation=modulation, noise=noise, plot = True) for sim in range(n_sims)]
        plv_grid[i_modulation, i_coupling] = np.mean(np.array(sims))
        plv_std[i_modulation, i_coupling] = np.std(np.array(sims))


#pickle save

# Compare low and high coupling
low_coupling = plv_grid[:,0]
high_coupling = plv_grid[:,4]
diff_plv = high_coupling - low_coupling

low_coupling_std = plv_std[:,0]
low_coupling_var = np.square(low_coupling_std)

high_coupling_std = plv_std[:,4]
high_coupling_var = np.square(high_coupling_std)

common_std = np.sqrt(low_coupling_var+high_coupling_var)

##### Figure ######
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FormatStrFormatter

fig = plt.figure()
plt.rcParams['font.size'] = '14'
gs = gridspec.GridSpec(ncols=2, nrows=2, height_ratios = [0.05, 1], width_ratios = [1.5,0.5]) 
gs.update(left=0.15, right = 0.95, bottom = 0.08, top = 0.90, wspace = 0.013, hspace = 0.07)

coupling_ticks = np.round(couplings, 3)
modulation_ticks = np.round(modulations, 3)

# Heatmap
ax1 = plt.subplot(gs[1,0])
plt1 = plt.imshow(plv_grid, interpolation='nearest', vmin=0, vmax=0.4,aspect='auto')
plt.xlabel('$Coupling\ in\ the\ θ\ band$',fontsize=18)
plt.ylabel('$Modulation\ of\ γ\ by\ θ$',fontsize=18)
ax1.set_xticks(range(n_coupling))
ax1.set_yticks(range(n_modulation))
ax1.set_xticklabels(coupling_ticks)
ax1.set_yticklabels(modulation_ticks)
plt.gca().invert_yaxis()

# Colorbar
cbax = plt.subplot(gs[0,0])
cb = Colorbar(ax=cbax, mappable = plt1, orientation='horizontal', ticklocation = 'top')
cb.set_label('PLV', labelpad = 10,fontsize=18)

# Line plot
ax2 = plt.subplot(gs[1,1])
y = range(diff_plv.shape[0])
plt.plot(diff_plv, y)
plt.fill_betweenx(y, diff_plv-common_std, diff_plv+common_std,alpha=.1)
ax2.set_xticks(range(diff_plv))
plt.xlabel('$ΔPLV_{High - Low}$',fontsize=18)
plt.axvline(x=0, color='k', ls='--')
ax2.set_yticks([])

fig.show()
#plt.savefig()
