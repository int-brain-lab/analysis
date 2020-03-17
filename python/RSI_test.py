"""
    Analysis of spike data in repeated sites
    Author: Sebastian Bruijns
"""

import numpy as np
import matplotlib.pyplot as plt
import brainbox as bb
import pandas as pd
import pickle
import itertools
import alf.io
from oneibl.one import ONE
from brainbox.io.one import load_spike_sorting
import seaborn as sns
import time


def calc_fr(spikes, clusters):
    temp = []
    for c in range(np.max(clusters)):
        if np.sum(clusters == c) == 0:
            continue
        s = spikes[clusters == c]
        fr = (s[-1] - s[0]) / len(s)
        temp.append(fr)
    return temp


def neuron_metric(metric, spikes, clusters):
    temp = []
    for c in range(np.max(clusters)):
        if np.sum(clusters == c) <= 1:
            continue
        s = spikes[clusters == c]
        m, _, _ = metric(s)
        temp.append(m)
    return temp


def metric_mean_fr(fr):
    return np.mean(fr)


def metric_max_fr(fr):
    return np.max(fr)


def metric_min_fr(fr):
    return np.min(fr)


def metric_mean_ff(spikes, clusters):
    x = neuron_metric(lambda x: bb.metrics.firing_rate_fano_factor(x, n_bins=1), spikes, clusters)
    return np.mean(x)


metric_funcs = [
    (metric_mean_fr, 'mean_firing_rate'),
    (metric_max_fr, 'max_firing_rate'),
    (metric_min_fr, 'min_firing_rate'),
    (metric_mean_ff, 'mean_fano_factor')
]

metric_funcs = [
    (metric_mean_ff, 'mean_fano_factor')
]

eids = ['4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', 'dda5fc59-f09a-4256-9fb5-66c67667a466']
labs = []
bad_eids = ['ecb5520d-1358-434c-95ec-93687ecd1396', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']

print("Will try to compare {} data sets".format(len(eids)))

probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]
one = ONE()

metrics = {}
for _, metric_name in metric_funcs:
    metrics[metric_name] = []

i = 0
eid = eids[0]
probe = probes[0]

spikes, _ = load_spike_sorting(eid, one=one)
spikes = spikes[0]

spikes, clusters = spikes[probe]['times'], spikes[probe]['clusters']


ts = spikes[clusters == 0]
hist_win=0.01
fr_win=0.5

t_tot = ts[-1] - ts[0]
n_bins_hist = np.int(t_tot / hist_win)
counts = np.histogram(ts, n_bins_hist)[0]
# Compute moving average of spike counts to get instantaneous firing rate in s.
n_bins_fr = np.int(t_tot / fr_win)
step_sz = np.int(len(counts) / n_bins_fr)
fr_slow = np.array([np.sum(counts[step:(step + step_sz)])
               for step in range(len(counts) - step_sz)]) / fr_win

fr_fast = np.convolve(counts, np.ones(step_sz)) / fr_win

@profile
def f(ts, hist_win, fr_win):
    t_tot = ts[-1] - ts[0]
    n_bins_hist = np.int(t_tot / hist_win)
    counts = np.histogram(ts, n_bins_hist)[0]
    # Compute moving average of spike counts to get instantaneous firing rate in s.
    n_bins_fr = np.int(t_tot / fr_win)
    step_sz = np.int(len(counts) / n_bins_fr)

    fr = np.array([np.sum(counts[step:(step + step_sz)])
                   for step in range(len(counts) - step_sz)]) / fr_win
    return fr

@profile
def f2(ts, hist_win, fr_win):
    t_tot = ts[-1] - ts[0]
    n_bins_hist = np.int(t_tot / hist_win)
    counts = np.histogram(ts, n_bins_hist)[0]
    # Compute moving average of spike counts to get instantaneous firing rate in s.
    n_bins_fr = np.int(t_tot / fr_win)
    step_sz = np.int(len(counts) / n_bins_fr)

    fr = np.convolve(counts, np.ones(step_sz)) / fr_win
    fr = fr[step_sz - 1:- step_sz]
    return fr

a = f(ts, hist_win, fr_win)
b = f2(ts, hist_win, fr_win)

print(a.shape)
print(b.shape)
print(np.sum(np.abs(a - b)))



quit()

def split(d, l):
    dizzle = {}
    for data, label, in zip(d, l):
        if label not in dizzle:
            dizzle[label] = [data]
        else:
            dizzle[label].append(data)
    return list(dizzle.values())


for i, (metric, metric_name) in enumerate(metric_funcs):

    data = metrics[metric_name]
    title = 'RSI_hist_' + metric_name
    list_data = split(data, labs)
    plt.hist(list_data, color=sns.color_palette("colorblind", len(list_data)), stacked=True)
    plt.title(title)
    plt.savefig('figures/hists/' + title)
    plt.show()
