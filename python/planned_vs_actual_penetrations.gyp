## Chris Krasniak 2020-10-08

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from oneibl.one import ONE
from sklearn.cluster import DBSCAN
from sklearn import mixture
import datajoint as dj
from ibl_pipeline import subject
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as ba
from matplotlib.lines import Line2D
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D



def get_all_traj_dj(provenence):

    """
    retrieves all trajectories, for a given provenence, options are "Micro-manipulator", "Planned",
    "Histology track", or "Ephys aligned histology track"
    adapted from Anne Urai's code 
    """
    ephys = dj.create_virtual_module('ephys', 'ibl_ephys')
    figpath = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

    traj = ephys.ProbeInsertion * (ephys.ProbeTrajectory & 'insertion_data_source = "{}"'.format(provenence)) \
            * subject.Subject * subject.SubjectLab
    traj = traj.proj('subject_nickname', 'x', 'y', 'theta', 'phi', 'theta', 'depth', 'lab_name',
                    'probe_label', session_date='DATE(session_start_time)')
    traj = traj.fetch(format='frame').reset_index()
    traj['probe_phi'] = traj['phi'].map({180:'180deg', 0:'0deg'})
    traj['angle'] = traj.theta

    traj['theta_name'] = traj['theta'].map({10:'10deg', 15:'15deg', 17:'17deg'})
    traj['probe_name'] = traj.probe_label 
    # traj['subject_nickname'] = traj.lab_name + ', ' + traj.subject_nickname
    traj = traj.sort_values(by=['subject_nickname'])
    traj['source'] = 'datajoint'
    traj_dj = traj.copy()
    return traj_dj

def get_all_traj_one(provenence):
    one = ONE()
    trajs = one.alyx.rest('trajectories', 'list', provenance=provenence)
    traj = pd.DataFrame(list(trajs))
    for kix, k in enumerate(traj.session[0].keys()):
        tmp_var = []
        for id, c in traj.iterrows():
            if k in c['session'].keys():
                tmp = c['session'][k]
            else:
                tmp = np.nan
            tmp_var.append(tmp)
            # also add the date
        traj[k] = tmp_var
        
    traj['theta_name'] = traj['theta'].map({10:'10deg', 15:'15deg', 17:'17deg'})
    traj['probe_name'] = traj.probe_name 
    traj['session_date'] = traj['start_time'].str[0:10]
    traj = traj.sort_values(by=['subject'])
    traj['source'] = 'one'
    traj['subject_nickname'] = traj['subject']
    traj_alyx = traj.copy()
    return traj_alyx

def probe_sph2cart(theta, phi, xyz0, depth=3.8):
    '''
    ported to python from needles
    takes spherical coordinates for a probe trajectory and returns cartesian coordinates for those
    trajectories in a n x 3 matrix where n is...
    theta : theta
    phi : phi
    xyz0 : xyz coordinates of the probe entering location in the brain
    depth: probe depth in mm
    '''
    xyz = np.zeros(3)
    xyz[0] = depth * np.sin(theta*np.pi/180) * np.cos(phi*np.pi/180) + xyz0[0]
    xyz[1] = depth * np.sin(theta*np.pi/180) * np.sin(phi*np.pi/180) + xyz0[1]
    xyz[2] = depth * np.cos(theta*np.pi/180) + xyz0[2]
    return xyz0, xyz

def dist(coords1,coords2):
    dist=0
    for i in range(len(coords1)):
        dist += (coords1[i]-coords2[i])**2
    dist=np.sqrt(dist)
    return dist

def coords2allen(traj,bregma=[228.5, 190]):
    traj['allenx'] = (traj.x * 1 / (pixelSize*1000)) + bregma[0]
    traj['alleny'] = (traj.y * -1 / (pixelSize*1000)) + bregma[1]
    traj['allenz'] = (traj.z * -1 / (voxelSize)) + bregma[1]
    return traj

bregma = [228.5, 190]
pixelSize = .025  # mm
allenOutline = np.load('/Users/ckrasnia/Desktop/Zador_Lab/scanData/allen_dorsal_outline')
bg_bregma = [540, 0, 570] #AP, DV, LR
voxelSize = 10 #um
brainGrid = np.load('/Users/ckrasnia/Desktop/Zador_Lab/IBL/lesion_project/brainGridData.npy')


all_traj = get_all_traj_one('Planned') #'Micro-manipulator'
# all_traj= all_traj[all_traj['lab']=='zadorlab']
labs = np.unique(all_traj.lab)
cmap2 = sns.color_palette("tab10")
labcolors = [cmap2[i] for i in range(len(labs))]

lab_data = {}


cnt=0
for lab in labs:

    fig, axs = plt.subplots(nrows=1,ncols=2,gridspec_kw={'width_ratios': [5, 1]})
    ax1 = axs[0]
    ax1.imshow(allenOutline, cmap="gray", interpolation='nearest')
    cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
    ax1.plot(bregma[0],bregma[1],'xk')
    ax1.axis('off')

    hist_traj = get_all_traj_one('Histology track')
    hist_traj = hist_traj[hist_traj['lab']==lab] 
    planned_traj = all_traj[all_traj['lab']==lab]
    hist_traj = coords2allen(hist_traj)
    planned_traj = coords2allen(planned_traj) 

    bregma = [228.5, 190]
    pixelSize = .025  # mm
    allenOutline = np.load('/Users/ckrasnia/Desktop/Zador_Lab/scanData/allen_dorsal_outline')
    bg_bregma = [540, 0, 570] #AP, DV, LR
    voxelSize = 10 #um
    brainGrid = np.load('/Users/ckrasnia/Desktop/Zador_Lab/IBL/lesion_project/brainGridData.npy')

    arrows = []
    for i in range(len(hist_traj)):
        probe = hist_traj.iloc[i]
        matching = planned_traj[planned_traj.probe_insertion==probe.probe_insertion]
        if not matching.empty:
            line = np.array([[matching.allenx.iloc[0],probe.allenx],[matching.alleny.iloc[0],probe.alleny]])
            arrow = np.array([[matching.allenx.iloc[0],matching.alleny.iloc[0]],[probe.allenx-matching.allenx.iloc[0],probe.alleny-matching.alleny.iloc[0]]])
            arrows.append(arrow)

    sum_x = 0
    sum_y = 0
    differences = []
    for arrow in arrows:
        length = np.sqrt(arrow[1,0]**2+arrow[1,1]**2)
        ax1.arrow(arrow[0,0],arrow[0,1],arrow[1,0],arrow[1,1],width=length/10,color=labcolors[cnt])
        differences.append(length*(pixelSize*1000))
    lab_data[lab] = differences
    
    ax2 = axs[1]
    ax2.bar([lab]*len(differences),np.nanmean(differences),alpha=.5,fill=False,edgecolor='k')
    ax2.scatter([lab]*len(differences),differences, color=labcolors[cnt])
    

    cnt+=1
plt.show(block=False)

ax2 = axs[1]
cnt=0

for name,data in lab_data.items():    
    ax2.bar([name]*len(data),np.nanmean(data),alpha=.5,fill=False,edgecolor='k')
    ax2.scatter([name]*len(data),data, color=labcolors[cnt])
    cnt+=1

ax2.set_ylabel('distance from planned (um)')
plt.xticks(rotation=45)
plt.show(block=False)





