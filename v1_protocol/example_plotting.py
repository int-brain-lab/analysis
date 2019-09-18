from pathlib import Path
import numpy as np
#from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
import numpy.ma as ma
import alf.io  
from brainbox.processing import bincount2D
import matplotlib.pyplot as plt
import ibllib.plots as iblplt
# define the path to the sessions we downloaded 
from pylab import *
ion()


main_path = Path('/home/mic/drive_codecamp/')
SES = {
    'A': main_path.joinpath(Path('ibl_witten_04/2019-08-04/002')), # RSC --> CA1 --> midbrain, good behavior, bad recroding
    'B': main_path.joinpath(Path('ibl_witten_04/2018-08-11/001')), # visual cortex, good behavior, noisy recording
    'C': main_path.joinpath(Path('KS005/2019-08-29/001')),  # left probe, bad behavior, good recording
    'D': main_path.joinpath(Path('KS005/2019-08-30/001')), # motor cortex, bad beahvior, good recording
    'E': main_path.joinpath(Path('ZM_1735/2019-08-01/001')), # activity in in red nucleaus, bad recording (serious lick artifacts and some units saturated) 
#    'F': main_path.joinpath(Path('KS005/2019-08-30/001')), # too large, didnt download for now
}
# select a session from the bunch
sid = 'E'
ses_path = Path(SES[sid])
# read in the alf objects
alf_path = ses_path / 'alf'
spikes = alf.io.load_object(alf_path, 'spikes')  
clusters = alf.io.load_object(alf_path, 'clusters')
channels = alf.io.load_object(alf_path, 'channels')
trials = alf.io.load_object(alf_path, '_ibl_trials')
wheel = alf.io.load_object(alf_path, '_ibl_wheel')

print(np.unique(clusters.probes))

T_BIN = 0.01 # time bin in sec
# compute raster map as a function of cluster number

probe_id=clusters.probes[spikes.clusters]

restrict = np.where(probe_id == 0)[0]

R, times, Clusters = bincount2D(spikes['times'][restrict], spikes['clusters'][restrict], T_BIN)

# Alternatively, order activity by cortical depth of neurons
d=dict(zip(spikes['clusters'][restrict],spikes['depths'][restrict]))
y=sorted([[i,d[i]] for i in d])
isort=np.argsort([x[1] for x in y])
R=R[isort,:]

# get trial number for each time bin       
trial_numbers = np.digitize(times,trials['goCue_times'])
print('Range of trials: ',[trial_numbers[0],trial_numbers[-1]])


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def add_stim_off_times(trials):
   on = 'stimOn_times'
   off = 'stimOff_times'
   trials[off] = np.zeros(shape=trials[on].shape)
   correct_trials = trials['feedbackType'] == 1
   trials[off][correct_trials] = trials['feedback_times'][correct_trials] + 1.0
   error_trials = trials['feedbackType'] == -1
   trials[off][error_trials] = trials['feedback_times'][error_trials] + 2.0

add_stim_off_times(trials)

def plot_trial(trial_number,R, times):

    ioff()
     
    a = list(trial_numbers) 
    first = a.index(trial_number) - 200 # add 2 sec 
    last  = len(a) - 1 - a[::-1].index(trial_number)

    ## Using rastermap defaults to order activity matrix
    ## by similarity of activity (requires R to contain floats)
    #model = rastermap.mapping.Rastermap().fit(R.astype(float))
    #isort = np.argsort(model.embedding[:, 0])
    #R = R[isort, :]

    plt.imshow(R[:,first:last], aspect='auto', cmap='binary', vmax=T_BIN / 0.001 / 4,
               extent=np.r_[times[[first, last]], Clusters[[0, -1]]], origin='lower')

    def restrict_timestamplist(q):
        
        start_idx = find_nearest(q, times[first])
        stop_idx = find_nearest(q, times[last])   
        if len(q[start_idx:stop_idx])>1:
            return q[start_idx:stop_idx][-1]
        else:
            return q[start_idx:stop_idx]
            

    iblplt.vertical_lines(restrict_timestamplist(trials['stimOn_times']), ymin=0, ymax=Clusters[-1], color='m',linewidth=0.5,label='stimOn_times')
 
    iblplt.vertical_lines(restrict_timestamplist(trials['feedback_times']), ymin=0, ymax=Clusters[-1], color='b',linewidth=0.5,label='feedback_times')

    iblplt.vertical_lines(restrict_timestamplist(trials['stimOff_times']), ymin=0, ymax=Clusters[-1], color='g',linewidth=0.5,label='stimOff_times')
    

    plt.xlabel('Time (s)')
    plt.ylabel('Cluster #; ordered by depth')
    plt.legend()
    
    plt.savefig('/home/mic/Rasters/%s/%s.svg' %(sid,trial_number))
    plt.close('all')
    plt.tight_layout()
    ion()
