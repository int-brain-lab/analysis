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
from sklearn.manifold import TSNE


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

eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199']
labs = []

print("Will try to compare {} data sets".format(len(eids)))

probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]
bad_eids = ['9a629642-3a9c-42ed-b70a-532db0e86199', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', 'ee40aece-cffd-4edb-a4b6-155f158c666a', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9']
one = ONE()

metrics = {}
for _, metric_name in metric_funcs:
    metrics[metric_name] = []


for i, (eid, probe) in enumerate(zip(eids, probes)):
    if eid in bad_eids: continue
    print(eid)
    spikes, _ = load_spike_sorting(eid, one=one)
    spikes = spikes[0]

    spikes, clusters = spikes[probe]['times'], spikes[probe]['clusters']


    times_stimon = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
    #times_feedback = one.load(eid, dataset_types=['trials.feedback_times'])[0]
    #feedback = one.load(eid, dataset_types=['trials.feedbackType'])[0]
    depths = np.array(one.load(eid, dataset_types=['clusters.depths']))

    session_path = one.path_from_eid(eid)
    if not session_path:
        print(session_path)
        print("no session path")
        continue

    _ = one.load(eid, dataset_types='clusters.metrics', download_only=True)

    try:
        _ = alf.io.load_object(session_path.joinpath('alf'), 'probes')
    except FileNotFoundError:
        print(session_path.joinpath('alf'))
        print("no probes")
        continue

    probe_path = session_path.joinpath('alf', probe)
    try:
        metrics = alf.io.load_object(probe_path, object='clusters.metrics')
    except FileNotFoundError:
        print(probe_path)
        print("one probe missing")
        continue


    quality = metrics.metrics.ks2_label == 'good'


    for d in depths:
        if d.shape[0] == np.max(np.unique(clusters)) + 1:
            depths = d
            break

    #times = times_feedback[feedback == 1]
    times = times_stimon

    start = time.time()
    a, b = bb.singlecell.calculate_peths(spikes, clusters, quality.index[quality], times)
    print(time.time() - start)


    depths = depths[quality]
    """vals, indizes = np.unique(clusters, return_index=True)
    clusts = [clusters[i] for i in sorted(indizes)]
    depths = depths[np.argsort(np.flip(clusts))]  # interesting results, weirdly enough"""

    perps = [5, 8]
    np.random.seed(4)
    for p in perps:
        neurons_embedded = TSNE(perplexity=p).fit_transform(a.means)
        plt.scatter(neurons_embedded[:, 0], neurons_embedded[:, 1], c=depths)
        title = "good sort Mouse {} Perplexity {}".format(one.list(eid, 'subject'), p)
        plt.title(title)
        plt.savefig('../../figures/' + title + '.png')
        plt.close()

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
