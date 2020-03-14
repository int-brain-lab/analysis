"""
    Analysis of general metrics in repeated sites
    Author: Sebastian Bruijns
"""

import numpy as np
import matplotlib.pyplot as plt
import brainbox as bb
from brainbox.quality.permutation_test import permut_test
import pandas as pd
import pickle
import itertools
import alf.io
from oneibl.one import ONE
import seaborn as sns

def metric_mean_fr(x): return np.mean(x['firing_rate'])


def metric_max_fr(x): return np.max(x['firing_rate'])


def metric_min_fr(x): return np.min(x['firing_rate'])


def metric_mean_ns(x): return np.mean(x['num_spikes'])


def metric_num_gu(x): return np.sum(x['ks2_label'] == 'good')


def metric_num_units(x): return len(x)


def metric_mean_ps(x): return np.mean(x['presence_ratio'])


metric_funcs = [
    (metric_mean_ns, 'mean_num_spikes'),
    (metric_num_gu, 'num_good_units'),
    (metric_num_units, 'num_units'),
    (metric_mean_ps, 'mean_presence_ratio')
]

eids = ['4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', 'dda5fc59-f09a-4256-9fb5-66c67667a466']
labs = []

print("Will try to compare {} data sets".format(len(eids)))

probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]
one = ONE()


metric_list = []
lab_list = []
for eid, probe in zip(eids, probes):

    session_path = one.path_from_eid(eid)
    if not session_path:
        print(session_path)
        print("no session path")
        continue

    _ = one.load(eid, dataset_types='clusters.metrics', download_only=True)

    try:
        probes = alf.io.load_object(session_path.joinpath('alf'), 'probes')
    except FileNotFoundError:
        print(session_path.joinpath('alf'))
        print("no probes")
        continue
    spikes = {}
    clusters = {}

    probe_path = session_path.joinpath('alf', probe)
    try:
        metrics = alf.io.load_object(probe_path, object='clusters.metrics')
    except FileNotFoundError:
        print(probe_path)
        print("one probe missing")
        continue

    labs.append(one.list(eid, 'labs'))
    metric_list.append(metrics.metrics)


def split(d, l):
    dizzle = {}
    for data, label, in zip(d, l):
        if label not in dizzle:
            dizzle[label] = [data]
        else:
            dizzle[label].append(data)
    return list(dizzle.values())

for i, (metric, metric_name) in enumerate(metric_funcs):

    data = [metric(x) for x in metric_list]
    print(data)
    title = 'RSI_hist_' + metric_name
    plt.hist(split(data, labs), color=sns.color_palette("colorblind", 6), stacked=True)
    plt.title(title)
    plt.savefig('figures/hists/' + title)
    plt.show()
