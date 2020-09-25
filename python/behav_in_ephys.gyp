import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from sklearn.cluster import DBSCAN
from sklearn import mixture
import datajoint as dj
from ibl_pipeline import subject
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as ba


def get_all_traj(provenence):

    """
    retrieves all trajectories, for a given provenence, options are "Micro-manipulator", "Planned",
    "Histology track", or "Ephys aligned histology track"
    adapted from Anne Urai's code 
    """
    ephys = dj.create_virtual_module('ephys', 'ibl_ephys')
    figpath = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

    traj = ephys.ProbeInsertion * (ephys.ProbeTrajectory & 'insertion_data_source = "{}"'.format(provenence)) \
            * subject.Subject 
    traj = traj.proj('subject_nickname', 'x', 'y', 'theta', 'phi', 'theta', 'depth',
                    'probe_label', session_date='DATE(session_start_time)')
    traj = traj.fetch(format='frame').reset_index()
    traj['probe_phi'] = traj['phi'].map({180:'180deg', 0:'0deg'})
    traj['angle'] = traj.theta

    traj['theta_name'] = traj['theta'].map({10:'10deg', 15:'15deg', 17:'17deg'})
    traj['probe_name'] = traj.probe_label + ', ' + traj.theta_name
    # traj['subject_nickname'] = traj.lab_name + ', ' + traj.subject_nickname
    traj = traj.sort_values(by=['subject_nickname'])
    traj['source'] = 'datajoint'
    traj_dj = traj.copy()
    return traj_dj


def get_craniotomies(penetrations):
    locs = np.asarray([penetrations.x,penetrations.y]).T
    new_locs = [tuple(row) for row in locs] 
    unique_locs,unique_idx = np.unique(new_locs,axis=0,return_index=True)
    penetrations = penetrations.iloc[unique_idx,:]
    clusters = DBSCAN(eps=1000, min_samples=1).fit_predict(unique_locs)
    num_cranios = max(clusters)
    ## find the center of the cranio just by getting the middle of all penetrations assigned to one cranio
    cranio_info = pd.DataFrame()
    subject_uuids = []
    first_recs = []
    centers = []
    sub_names = []
    for cranio in range(num_cranios):
        idx = np.where(clusters==cranio)[0]
        new_locs = unique_locs[idx]
        centers.append(np.mean(new_locs,axis=0))
        first_recs.append(min(penetrations.session_start_time[idx]))
        subject_uuids.append(penetrations.subject_uuid[0])
        sub_names.append(penetrations.subject_nickname[0])
    cranio_info['subject_uuid'] = subject_uuids
    cranio_info['first_recs'] = first_recs
    cranio_info['centers'] = centers
    cranio_info['subject_nickname'] = sub_names
    return cranio_info
    
def plot_biased_psychos(ses_info,before):
    if before:
        colors = ['blue','green','red']
    else:
        colors = ['darkblue','darkgreen','darkred']
    x = np.arange(-1,1,.01)
    cnt = 0
    plt.figure()
    for ind,block in ses_info.iterrows():
        pars = [block.threshold,block.bias,block.lapse_low,block.lapse_high]
        fit = psy.erf_psycho_2gammas(pars,x)
        plt.plot(x,fit,'-',color=colors[cnt%3])
        cnt+=1
    plt.xlabel['Signed Contrast']
    plt.ylabel['Fraction Choose Right']
    plt.show()


traj = get_all_traj('Micro-manipulator')
subjects = np.unique(traj.subject_uuid)
cranios = pd.DataFrame(columns=['subject_uuid','first_recs','centers'])
for i,sub in enumerate(subjects):
    pens = traj[traj.subject_uuid==sub] 
    days = np.unique(pens.session_date) #some penetrations are marked twice on the same day, want to filter these out
    column_names = pens.keys()
    use_pens = pd.DataFrame(columns=column_names)
    for day in days:
        use_pens=pd.concat([use_pens,pd.DataFrame(pens.iloc[max(loc for loc, val in enumerate(days) if val == day),:]).T],ignore_index=True)
    craniotomies = get_craniotomies(use_pens)
    cranios = pd.concat([cranios,craniotomies])
cranios.reset_index(inplace=True)

behav_before = []
behav_after = []
for cranio in range(len(cranios)):
    c_info = cranios.iloc[cranio,:]
    q = ba.PsychResultsBlock & (subject.Subject & 'subject_nickname="{}"'.format(cranios.subject_nickname[cranio]))
    if not q:
        continue # if there's no behavior (probably a certification mouse)
    # before = q & 'session_start_time < "{}"'.format(cranios.first_recs[cranio])
    sub_psych = pd.DataFrame(q.fetch(as_dict=True))
    cranio_time = c_info.first_recs
    cranio_ses = sub_psych[sub_psych.session_start_time == c_info.first_recs] 
    #### here if I want I could get more sessions after this cranio but before the next one
    before = sub_psych[sub_psych.session_start_time < c_info.first_recs]
    last3before = before.iloc[-9:,:]
    if len(np.unique(last3before.session_start_time)) != 3:
        continue
        # if these 9 sessions aren't for 3 days, then skip this cranio
    # if c_info[0] == 0: 
    #     # this will save behav_before for extra loops if a mouse has multiple cranios, so this will
    #     # compare behavior before any cranios to behavior after a given cranio (could be multiple)
    #     keep_before = last3before
    
    behav_before.append(last3before)
    behav_after.append(cranio_ses)

for i in range(len(behav_before)):
    print(i)
    if behav_after[i].empty:
        plot_biased_psychos(behav_before[i],True)
        plot_biased_psychos(behav_after[i],False)