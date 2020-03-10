#%%
from ibl_pipeline import subject, acquisition
import pandas as pd
from oneibl.one import ONE
import alf.io
from pathlib import Path
import os
import numpy as np
from operator import add
import matplotlib.pyplot as plt
import seaborn as sns
from uuid import UUID

one = ONE()

#all_data = (subject.Subject * (subject.SubjectLab & ['lab_name = "mrsicflogellab"', 'lab_name = "hoferlab"']) * 
#            (acquisition.Session & 'task_protocol LIKE "%ephys%"')).fetch(format='array')
all_data = (subject.Subject * (subject.SubjectLab & ['lab_name = "zadorlab"']) * 
            (acquisition.Session & 'task_protocol LIKE "%ephys%"')).fetch(format='array')
all_data = pd.DataFrame(all_data)
all_data = all_data.sort_values('session_start_time').reset_index(drop=True)

#Angelaki lab need to drop one session
#all_data = all_data.drop(index = np.where(all_data.session_uuid == UUID('3bbabd25-0990-4c20-95f8-65888acc873b'))[0])

#Churchland lab need to drop a couple of sessions
#all_data = all_data.drop(index=np.where(all_data.subject_nickname == 'CSHL_007')[0]).reset_index(drop=True)
#all_data = all_data.drop(index = np.where(all_data.session_uuid == UUID('63b83ddf-b7ea-40db-b1e2-93c2a769b6e5'))[0]).reset_index(drop=True)

d_types = ['clusters.metrics', 'probes.description']

dates = []
probe_ID = []
probe_N = []
n_good = []
n_mua = []
subjects = []
n_total = []

session_info = pd.DataFrame(columns = ['subject', 'date', 'session_number', 'file_exists', 'probe_00', 'probe_01'], index = np.arange(len(all_data)))

#%%
for iS, __ in all_data.iterrows():
    print(iS)
    date = all_data.session_start_time[iS].strftime('%Y-%m-%d')
    subject = all_data.subject_nickname[iS]
    session_info.subject[iS] = subject
    session_info.date[iS] = date
    eid = one.search(subject=subject, date=date)
    if len(eid) > 1:
        sess_id = eid[len(eid) - 1]
        #print('More than one session')
        files = one.load(sess_id, dataset_types=d_types, clobber=False, download_only=True)
        session_info.session_number[iS] = len(eid)
        if files[0] is None:
            sess_id = eid[0]
            files = one.load(sess_id, dataset_types=d_types, clobber=False, download_only=True)
            session_info.session_number[iS] = len(eid)
    elif len(eid) == 1:
        sess_id = eid[0]
        files = one.load(sess_id, dataset_types=d_types, clobber=False, download_only=True)
        session_info.session_number[iS] = len(eid)


    if files[0] is None:
        session_info.file_exists[iS] = False
        session_info.probe_00[iS] = False
        session_info.probe_01[iS] = False
    else:
        session_info.file_exists[iS] = True
        session_info.probe_00[iS] = False
        session_info.probe_01[iS] = False

        info = alf.io.load_file_content(files[-1])
        alf_path = os.path.split(files[-1])[0]

        for iP in info:
            probe = iP.get('label')
            probe_path = Path(alf_path, probe)

            if probe == 'probe00': session_info.probe_00[iS] = True  
            if probe == 'probe01': session_info.probe_01[iS] = True


            if Path.exists(probe_path):
                clusters = alf.io.load_object(probe_path, 'clusters')
                subjects.append(subject + '__')
                dates.append(date)
                probe_ID.append(iP.get('serial'))
                probe_N.append(probe)
                n_good.append(len(np.where(clusters.metrics.ks2_label == 'good')[0]))
                n_mua.append(len(np.where(clusters.metrics.ks2_label == 'mua')[0]))
                n_total.append(len(clusters.metrics.ks2_label))

    
#%%

probe_data = pd.DataFrame(columns = ['session', 'probe_ID', 'probe_N', 'n_good', 'n_mua', 'n_total'])
probe_data.session = list(map(add, subjects, dates))
probe_data.probe_ID = probe_ID
probe_data.probe_N = probe_N
probe_data.n_good = n_good
probe_data.n_mua = n_mua
probe_data.n_total = n_total

#%%

probes, n_probes = np.unique(probe_data.probe_ID, return_counts=True)
probes = probes[np.argsort(n_probes)]

n_probes = len(probes)

#%%
f, axis = plt.subplots(1, n_probes, figsize=(18,5), sharex=True, sharey = True)
for iP, pr in enumerate(probes):
    idx = np.where(probe_data.probe_ID == pr)[0]
    sub_data = probe_data.loc[idx]
    sns.set_color_codes('pastel')
    sns.barplot(x = 'n_total', y = 'session', data = sub_data, label = 'mua',  color = 'y', edgecolor = 'w', ax = f.axes[(n_probes - 1)- iP])
    g = sns.barplot(x = 'n_good', y = 'session', data = sub_data, label = 'good',  color = 'g', edgecolor = 'w', ax = f.axes[(n_probes - 1) -iP])
    if iP == (n_probes - 1):
        ax = f.axes[-1]
        ax.legend(ncol = 1, loc = 'center right', bbox_to_anchor=(1.5, 0.5))
        g.set_ylabel('session_no', fontsize=10)
    else:
        g.set_ylabel('')

    sns.despine(left = True, bottom = True)

    g.set_title('Probe ' f"{pr}", fontsize = 10)
    g.set_xlabel('no. of neurons', fontsize = 10)
  

    g.set_yticklabels(np.arange(len(sub_data)))
 
plt.show()

