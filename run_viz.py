#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : run_viz.py
# description     : Load and plot the results of the parallel simulations
# author          : Guillaume Dumas, Quentin Moreau
# date            : 2022-08-27
# version         : 1
# usage           : python run_viz.py
# notes           : require to run first run_kuramoto_parallel.py
# python_version  : 3.7-3.9
# ==============================================================================

import numpy as np
from scipy.signal import hilbert
from kuramoto import Kuramoto
from joblib import Parallel, delayed
import time
from statsmodels.stats.weightstats import ztest
import sys

n_coupling = 11
n_modulation = 11
n_sims = 10000
noise = 0.6
z_threshold = 1.96
# Init values ranges & grid
couplings = np.linspace(0, 1, n_coupling)
modulations = np.linspace(0, 1, n_modulation)
coupling_grid, modulation_grid = np.meshgrid(couplings, modulations, sparse=False, indexing='xy')

plv_grid_par = np.load('plv_grid.npy')
common_std = np.load(f'{n_sims} Sims with a {noise} noise_STD.npy')
SNR_avg = np.load(f'{n_sims} Sims with a {noise} noise_SNR.npy')
zvals = np.load(f'{n_sims} Sims with a {noise} noise_zvals.npy')
pvals = np.load(f'{n_sims} Sims with a {noise} noise_pvals.npy')

# PLV Difference between high and low coupling for Figure
low_coupling = plv_grid_par[:, 0]
high_coupling = plv_grid_par[:, n_coupling - 1]
diff_plv = high_coupling - low_coupling

##### Figure ######
from turtle import color
from matplotlib.colorbar import Colorbar
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FormatStrFormatter


fig = plt.figure()
plt.rcParams['font.size'] = '14'
gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios = [0.05, 1],  width_ratios = [1.5,0.5,0.25])
gs.update(left=0.10, right = 0.97, bottom = 0.1, top = 0.90, wspace = 0.013, hspace = 0.07)

coupling_ticks = np.round(couplings, 3)
modulation_ticks = np.round(modulations, 3)

# Heatmap
ax1 = plt.subplot(gs[1,0])
plt1 = plt.imshow(plv_grid_par, interpolation='nearest', vmin=0, vmax=0.4,aspect='auto')
plt.xlabel('Inter-brain Coupling in θ',fontsize=18)
plt.ylabel('Intra-brain θ-γ CFC',fontsize=18)
ax1.set_xticks(range(n_coupling))
ax1.set_yticks(range(n_modulation))
ax1.set_xticklabels(coupling_ticks)
ax1.set_yticklabels(modulation_ticks)
plt.gca().invert_yaxis()


# Colorbar
cbax = plt.subplot(gs[0,0])
cb = Colorbar(ax=cbax, mappable = plt1, orientation='horizontal', ticklocation = 'top')
cb.set_label('Inter-brain γ-PLV', labelpad = 10,fontsize=18)

# Line plot
ax2 = plt.subplot(gs[1,1])
y = range(diff_plv.shape[0])
plt.plot(diff_plv, y, color='mediumblue')
plt.fill_betweenx(y, diff_plv-common_std, diff_plv+common_std,alpha=1, color ='lavender')
ax2.set_xticks([0, 0.15])
ax2.set_title('High - Low coupling',fontsize=18)
plt.xlabel('$ΔPLV$',fontsize=16)
plt.axvline(x=0, color='k', ls='--')
ax2.set_yticks([])

y1 = y

# zvals histogram
ax3 = plt.subplot(gs[1,2])
clrs = ['lavender' if (x < z_threshold) else 'mediumblue' for x in zvals]
plt.barh(y1, width= zvals, color=clrs)
ax3.set_yticks([])
ax3.set_title('$Z test$',fontsize=18)
plt.axvline(x=z_threshold, color='k', ls='--')
plt.text(4,1,'Significance\n threshold',rotation=0, fontsize = 10)
plt.xlabel('$Z values$',fontsize=16)

fig.show()
plt.tight_layout()
fig = plt.gcf()  # get current figure
fig.set_size_inches(15, 10)
plt.savefig(f'{n_sims} Sims with a {noise} noise NEW.pdf', bbox_inches='tight')
plt.savefig(f'{n_sims} Sims with a {noise} noise NEW.png', bbox_inches='tight')
