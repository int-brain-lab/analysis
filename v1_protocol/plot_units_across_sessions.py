import os
from pathlib import Path
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from oneibl.one import ONE
import alf.io as aio
import brainbox as bb
​
# get dataset names
dtypes = [
        'clusters.amps',
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'clusters.peakToTrough',
        'clusters.uuids',
        'clusters.waveforms',
        'clusters.waveformsChannels',
        'spikes.amps',
        'spikes.clusters',
        'spikes.depths',
        'spikes.samples',
        'spikes.templates',
        'spikes.times'
        ]
​
one = ONE()
​
eid0 = one.search(subject='SWC_026', date='2019-09-16', number=1)[0]
eid1 = one.search(subject='KS003', date='2019-11-19', number=1)[0]
eid2 = one.search(subject='lic3', date='2019-08-27', number=2)[0]
eid3 = one.search(subject='cer-5', date='2019-10-25', number=1)[0]
eid4 = one.search(subject='CSHL_020', date='2019-12-04', number=5)[0]
eid5 = one.search(subject='ZM_2406', date='2019-11-12', number=1)[0]
eid6 = one.search(subject='CSK-scan-008', date='2019-12-09', number=8)[0]
eid7 = one.search(subject='NYU-18', date='2019-10-23', number=1)[0]
​
eids = [eid0, eid1, eid2, eid3, eid4, eid5, eid6, eid7]
​
n_units = np.zeros((len(eids),))
n_good_units = np.zeros((len(eids),)) 
​
for i, eid in enumerate(eids):
    # load data to disk
    d_paths = one.load(eid, dataset_types=dtypes, clobber=False, download_only=True)
    
    # get paths and load data in memory
    alf_probe_path = os.path.split(d_paths[0])[0]
    spks_b = aio.load_object(alf_probe_path, 'spikes')
    units_b = bb.processing.get_units_bunch(spks_b)
    
    # filter units and get nubmer of good units and number of total units
    T = spks_b.times[-1] - spks_b.times[0]  # length of recording session
    filt_units = bb.processing.filter_units(units_b, T, min_amp=50e-6, min_fr=1)
    
    # get num_units & num_good units
    n_units[i] = np.max(spks_b.clusters)
    n_good_units = len(filt_units)
​
# 2 group bar plot(all units, good units)
names = ['swc026_09-16_1', 'ks003_11-19_1', 'lic3_08-27_2', 'cer5_10-25_1', 'CSHL020_12-04_5',
         'ZM2406_11-12_1', 'CSKscan008_12-09_1', 'NYU18_10-23_1']
​
units_eid0 = [n_units[0], n_good_units[0]]
units_eid1 = [n_units[1], n_good_units[1]]
units_eid2 = [n_units[2], n_good_units[2]]
units_eid3 = [n_units[3], n_good_units[3]]
units_eid4 = [n_units[4], n_good_units[4]]
units_eid5 = [n_units[5], n_good_units[5]]
units_eid6 = [n_units[6], n_good_units[6]]
units_eid7 = [n_units[7], n_good_units[7]]
​
​
barWidth = 0.25
r0 = np.arange(len(n_units))
r1 = [x + barWidth for x in r0]
​
plt.bar(r0, n_units, width=barWidth, label='n_units')
plt.bar(r1, n_good_units, width=barWidth, label='n_good_units')
plt.xticks([r + barWidth for r in range(len(n_units))], [names[0], names[1], names[2], names[3],
                                                         names[4], names[5], names[6], names[7]])
plt.legend()
