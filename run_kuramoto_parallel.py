#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : run_kuramoto_parallel.py
# description     : Demonstrates the link between crossfrequency coupling & IBC
# author          : Guillaume Dumas, Quentin Moreau
# date            : 2021-11-09
# version         : 2
# usage           : python run_kuramoto_parallel.py
# notes           : require kuramoto.py (version by D. Laszuk)
# python_version  : 3.7-3.9
# ==============================================================================

import numpy as np
from scipy.signal import hilbert
from kuramoto import Kuramoto
from joblib import Parallel, delayed
import time
from statsmodels.stats.weightstats import ztest
import sys


def simu(coupling=0.1, modulation=0.1, noise=1):
    low_freq_sd = 1
    low_freq_mean = 6
    high_freq_mean = 40

    # Defining time array
    t0, t1, dt = 0, 60, 0.01
    T = np.arange(t0, t1, dt)

    # Y0, W, K are initial phase, intrinsic freq and
    # coupling K matrix respectively
    Y0 = np.random.rand(2) * 2 * np.pi
    W = 2 * np.pi * np.array(np.random.randn(2) * low_freq_sd + low_freq_mean)

    W12 = coupling
    W21 = coupling
    K1 = np.array([[0, W12],
                   [W12, 0]])
    K2 = np.array([[0, W21],
                   [W21, 0]])

    K = np.dstack((K1, K2)).T

    # Passing parameters as a dictionary
    init_params = {'W': W, 'K': K, 'Y0': Y0, 'noise': 'uniform'}

    # Running Kuramoto model
    kuramoto = Kuramoto(init_params)
    odePhi = kuramoto.solve(T)

    # Computing low and high rhythms
    low_fb = np.sin(odePhi)
    high_fb = np.vstack((np.sin(T * 2 * np.pi * high_freq_mean) * ((1 - modulation) + (modulation * low_fb[0])) + np.random.randn(odePhi.shape[1]) * noise,
                         np.sin(T * 2 * np.pi * high_freq_mean) * ((1 - modulation) + (modulation * low_fb[1])) + np.random.randn(odePhi.shape[1]) * noise))

    # Separate Signal and Noise
    signal_osc1 = np.sin(T * 2 * np.pi * high_freq_mean)
    noise_osc1 = np.random.randn(odePhi.shape[1]) * noise

    # Extract signal envelope
    signal = np.mean(pow(signal_osc1, 2))
    noise = np.mean(pow(noise_osc1, 2))

    # Compute Signal to Noise ratio
    SNR = signal / noise
    SNR_db = 20 * np.log(SNR)

    high_phase = np.angle(hilbert(high_fb))

    # Extract PLV
    PLV = np.abs(np.mean(np.exp(1j * (high_phase[1] - high_phase[0]))))
    return PLV, SNR_db


# Simulation Parameters
n_coupling = 11
n_modulation = 11
n_sims = 10000
noise = 0.6

# Parallel Processing
n_jobs = 40

# Init values ranges & grid
couplings = np.linspace(0, 1, n_coupling)
modulations = np.linspace(0, 1, n_modulation)
coupling_grid, modulation_grid = np.meshgrid(couplings, modulations, sparse=False, indexing='xy')

# Run simulations Parallel Processing for Compute Canada
plv_grid_par = np.zeros(coupling_grid.shape) * np.nan
plv_std_par = np.zeros(coupling_grid.shape) * np.nan


def run_simulations_par(i_coupling, i_modulation):
    start_time = time.time()
    stream = getattr(sys, "stdout")
    print(f"Running simulations for Coupling={i_coupling} and Modulation={i_modulation}", file=stream)
    coupling = coupling_grid[i_modulation, i_coupling]
    modulation = modulation_grid[i_modulation, i_coupling]
    tmp = [simu(coupling=coupling, modulation=modulation, noise=noise) for sim in range(n_sims)]
    print(f'Elapsed time for the entire processing: {time.time() - start_time} s')
    stream.flush()
    return [sim[0] for sim in tmp], [sim[1] for sim in tmp]


print('Starting simulations...')
start = time.time()
Outputs = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(delayed(run_simulations_par)(i_coupling, i_modulation)
                                                              for i_coupling in range(n_coupling)
                                                              for i_modulation in range(n_modulation))
stop = time.time()
elapsed = stop - start
print(f'Elapsed time for the entire processing: {elapsed} s')

# Outputs from Paralell
Outputs_arr = np.asarray(Outputs)
PLV = Outputs_arr[:, 0, :]
SNR = Outputs_arr[:, 1, :]
SNR_avg = np.mean(SNR)
# print(f'The SNR with a noise of {noise} is {SNR_avg} dB')

PLV_reshape = PLV.reshape(n_coupling, n_modulation, n_sims)
plv_grid_par = np.mean(PLV_reshape, 2).T
plv_std_par = np.std(PLV_reshape, 2).T

# Compare low and high coupling for stats
low_coupling_stat = PLV_reshape[0, :, :]
high_coupling_stat = PLV_reshape[n_coupling - 1, :, :]
diff_plv_stat = high_coupling_stat - low_coupling_stat

# One sample Z test
zvals, pvals = ztest(diff_plv_stat.T, value=0, alternative='larger')
z_threshold = 1.96

# PLV Difference between high and low coupling for Figure
low_coupling = plv_grid_par[:, 0]
high_coupling = plv_grid_par[:, n_coupling - 1]
diff_plv = high_coupling - low_coupling

low_coupling_std = plv_std_par[:, 0]
high_coupling_std = plv_std_par[:, n_coupling - 1]
common_std = np.mean(low_coupling_std + high_coupling_std)

np.save('plv_grid', plv_grid_par)
np.save(f'{n_sims} Sims with a {noise} noise_STD', common_std)
np.save(f'{n_sims} Sims with a {noise} noise_SNR', SNR_avg)
np.save(f'{n_sims} Sims with a {noise} noise_zvals', zvals)
np.save(f'{n_sims} Sims with a {noise} noise_pvals', pvals)