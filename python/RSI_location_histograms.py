import numpy as np
from brainbox.io.one import load_spike_sorting, load_channel_locations
from oneibl.one import ONE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


one = ONE(base_url="https://dev.alyx.internationalbrainlab.org")
eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199']
probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]

# eids that actively crash the code
bad_eids = ['ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']
# only eids that have data (histology) so far
good_eids = ['aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919']
names_n_counts = []
names = []
all_data = []

for i, (eid, probe) in enumerate(zip(eids, probes)):
    if eid not in good_eids:
        print('skipped')
        continue
    if eid == '614e1937-4b24-4ad3-9055-c8253d089919':
        probe = 'probe00'
    print(eid)
    print(one.list(eid, 'subjects'))
    print(probe)
    channels = load_channel_locations(eid, one=one)
    # spikes, clusters = load_spike_sorting(eids[0], one=one)
    all_data.append(channels[probe])
    names_n_counts.append(channels[probe].acronym)
    names.append(one.list(eid, 'subjects'))

    print(len(channels[probe].acronym))


fs = 18

for i, d in enumerate(names_n_counts):
    df = pd.DataFrame(data=d, columns=["acronym"])
    ax = plt.subplot(2, 3, i + 1)
    plt.title(names[i], fontsize=fs)
    sns.countplot(y='acronym', data=df, orient='v')
    if i < 3:
        plt.xlabel(None)
    else:
        plt.xlabel('Site counts', fontsize=fs)
    if i % 3 != 0:
        plt.ylabel(None)
    else:
        plt.ylabel('Brain region', fontsize=fs)
plt.show()
