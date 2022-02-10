#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : run_kuramoto_parallel.py
# description     : Demonstrates the link between crossfrequency coupling & IBS
# author          : Guillaume Dumas, Quentin Moreau
# date            : 2021-11-09
# version         : 1
# usage           : python run_kuramoto_parallel.py
# notes           : require kuramoto.py (version by D. Laszuk)
# python_version  : 3.7-3.9
# ==============================================================================

from matplotlib.colorbar import Colorbar
import numpy as np
from scipy.signal import hilbert
import pylab as plt
from kuramoto import Kuramoto
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time

def simu(coupling=0.1, modulation=0.1, noise=0.1):
    low_freq_sd = 1
    low_freq_mean = 6

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

    high_phase = np.angle(hilbert(high_fb))
    return np.abs(np.mean(np.exp(1j * (high_phase[1] - high_phase[0]))))


# Simulation Parameters
n_coupling = 11
n_modulation = 11
n_sims = 5000
noise = 0.7

# Parallel Processing
n_jobs = 8

# Init values ranges & grid
couplings = np.linspace(0, 1, n_coupling)
modulations = np.linspace(0, 1, n_modulation)
coupling_grid, modulation_grid = np.meshgrid(couplings, modulations, sparse=False, indexing='xy')

# Run simulations Parallel Processing for Compute Canada
plv_grid_par = np.zeros(coupling_grid.shape) * np.nan
plv_std_par = np.zeros(coupling_grid.shape) * np.nan

def run_simulations_par(i_coupling, i_modulation):
                coupling = coupling_grid[i_modulation, i_coupling]
                modulation = modulation_grid[i_modulation, i_coupling]
                sims = [simu(coupling=coupling, modulation=modulation, noise=noise) for sim in range(n_sims)]

                return sims #plv_grid_par, plv_std_par

start = time.time()
PLV = Parallel(n_jobs = n_jobs, verbose=10)(delayed(run_simulations_par)(i_coupling, i_modulation) for i_coupling in range(n_coupling) for i_modulation in range(n_modulation))
stop = time.time()
print('\nParallel processing')
print('Elapsed time for the entire processing: {:.2f} s'.format(stop - start))

# Outputs from Paralell
PLV_arr = np.asarray(PLV)
PLV_reshape = PLV_arr.reshape(n_coupling,n_modulation,n_sims)
plv_grid_par = np.mean(PLV_reshape,2).T
plv_std_par = np.std(PLV_reshape, 2).T

# Compare low and high coupling
low_coupling = plv_grid_par[:,0]
high_coupling = plv_grid_par[:,n_coupling-1]
diff_plv = high_coupling - low_coupling

low_coupling_std = plv_std_par[:,0]
#low_coupling_var = np.square(low_coupling_std)

high_coupling_std = plv_std_par[:,n_coupling-1]
#high_coupling_var = np.square(high_coupling_std)

common_std = np.mean(low_coupling_std+high_coupling_std)

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
plt1 = plt.imshow(plv_grid_par, interpolation='nearest', vmin=0, vmax=0.4,aspect='auto')
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
# ax2.set_xticks(range(diff_plv.shape[1]))
plt.xlabel('$ΔPLV_{High - Low}$',fontsize=18)
plt.axvline(x=0, color='k', ls='--')
ax2.set_yticks([])

fig.show()
plt.savefig(f'{n_sims} Sims with a {noise} noise.pdf')
np.save('plv_grid', plv_grid_par)
