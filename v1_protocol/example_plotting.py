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
from oneibl.one import ONE

ion()

# Ephys data can be downloaded via ONE at some point?

## get the data from flatiron and the current folder
#one = ONE()
#eid = one.search(subject='ZM_1735', date='2019-08-01', number=1)
#D = one.load(eid[0], clobber=False, download_only=True)
#ses_path = Path(D.local_path[0]).parent

## read in the alf objects
#alf_path = ses_path / 'alf'

# data is here: http://ibl.flatironinstitute.org/codecamp/
alf_path = '/home/mic/Downloads/FlatIron/ZM_1735_2019-08-01_001/ZM_1735/2019-08-01/001/alf'

spikes = alf.io.load_object(alf_path, 'spikes')  
clusters = alf.io.load_object(alf_path, 'clusters')
channels = alf.io.load_object(alf_path, 'channels')
trials = alf.io.load_object(alf_path, '_ibl_trials')
#wheel = alf.io.load_object(alf_path, '_ibl_wheel')

T_BIN = 0.01 # time bin in sec
# compute raster map as a function of cluster number

# just get channels from probe 0, as there are two probes here
probe_id=clusters['probes'][spikes['clusters']]
restrict = np.where(probe_id == 0)[0]

R, times, Clusters = bincount2D(spikes['times'][restrict], spikes['clusters'][restrict], T_BIN)

# Order activity by cortical depth of neurons
d=dict(zip(spikes['clusters'][restrict],spikes['depths'][restrict]))
y=sorted([[i,d[i]] for i in d])
isort=np.argsort([x[1] for x in y])
R=R[isort,:]

# get trial number for each time bin       
trial_numbers = np.digitize(times,trials['goCue_times'])
print('Range of trials: ',[trial_numbers[0],trial_numbers[-1]])

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

    '''
    Plot a rasterplot for a given trial, ordered by insertion depth, with 
    'stimOn_times','feedback_times' and 'stimOff_times' 
    '''
     
    a = list(trial_numbers) 
    first = a.index(trial_number)
    last  = len(a) - 1 - a[::-1].index(trial_number)

    plt.imshow(R[:,first:last], aspect='auto', cmap='binary', vmax=T_BIN / 0.001 / 4,
               extent=np.r_[times[[first, last]], Clusters[[0, -1]]], origin='lower')

    def restrict_timestamplist(q): 
        
        l = []
        for i in q:
            if i > times[first] and i < times[last] :
                l.append(i)
        return l  

    iblplt.vertical_lines(restrict_timestamplist(trials['stimOn_times']), ymin=0, ymax=Clusters[-1], color='m',linewidth=0.5,label='stimOn_times')
 
    iblplt.vertical_lines(restrict_timestamplist(trials['feedback_times']), ymin=0, ymax=Clusters[-1], color='b',linewidth=0.5,label='feedback_times')

    iblplt.vertical_lines(restrict_timestamplist(trials['stimOff_times']), ymin=0, ymax=Clusters[-1], color='g',linewidth=0.5,label='stimOff_times')
    

    plt.xlabel('Time (s)')
    plt.ylabel('Cluster #; ordered by depth')
    plt.legend()
    
    #plt.savefig('/home/mic/Rasters/%s.svg' %(trial_number))
    #plt.close('all')
    plt.tight_layout()

#Get a raster plot
if __name__ == "__main__":
    # get a raster plot for a particular trial 
    plot_trial(235,R, times) 




######################### Some example plot using this data? ##

spike_times = spikes['times'][restrict] 
spike_clusters = spikes['clusters'][restrict]


def bin_responses(spike_times, spike_clusters, stim_times, stim_values):
    """
    Compute spike counts during grating presentation
    :param spike_times: array of spike times
    :type spike_times: array-like
    :param spike_clusters: array of cluster ids associated with each entry in `spike_times`
    :type spike_clusters: array-like
    :param stim_times: stimulus presentation times; array of size (M, 2) where M is the number of
        stimuli; column 0 is stim onset, column 1 is stim offset
    :type stim_times: array-like
    :param stim_values: grating orientations in radians
    :type stim_values: array-like
    :return: number of spikes for each clusterduring stimulus presentation
    :rtype: array of shape `(n_clusters, n_stims, n_stim_reps)`
    """
    cluster_ids = np.unique(spike_clusters)
    n_clusters = len(cluster_ids)
    stim_ids = np.unique(stim_values)
    n_stims = len(stim_ids)
    n_reps = len(np.where(stim_values == stim_values[0])[0])
    responses = np.zeros(shape=(n_clusters, n_stims, n_reps))
    stim_reps = np.zeros(shape=n_stims, dtype=np.int)
    for stim_time, stim_val in zip(stim_times, stim_values):
        i_stim = np.where(stim_ids == stim_val)[0][0]
        i_rep = stim_reps[i_stim]
        stim_reps[i_stim] += 1
        # filter spikes
        idxs = (spike_times > stim_time[0]) & \
               (spike_times <= stim_time[1]) & \
               np.isin(spike_clusters, cluster_ids)
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]
        # bin spikes similar to bincount2D: x = spike times, y = spike clusters
        bin_size = np.diff(stim_time)[0]
        xscale = stim_time
        xind = (np.floor((i_spikes - stim_time[0]) / bin_size)).astype(np.int64)
        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)
        # store
        bs_idxs = np.isin(cluster_ids, yscale)
        responses[bs_idxs, i_stim, i_rep] = r[:, 0]
    return responses


def compute_selectivity(means, thetas, measure):
    """
    Compute direction or orientation selectivity index measure, as well as preferred direction/ori
    OSI/DSI calculated as in Mazurek et al 2014:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4123790/
    :param means: mean responses over multiple stimulus presentations; array of shape
        `(n_clusters, n_stims)`
    :type means: array-like
    :param thetas: grating direction in radians (corresponds to dimension 1 of `means`)
    :type thetas: array-like
    :param measure: 'ori' | 'dir' - compute orientation or direction selectivity
    :type measure: str
    :return: tuple (index, preference)
    """
    if measure == 'dir' or measure == 'direction':
        factor = 1
    elif measure == 'ori' or measure == 'orientation':
        factor = 2
    else:
        raise ValueError('"%s" is an invalid measure; must be "dir" or "ori"' % measure)
    vector_norm = means / np.sum(means, axis=1, keepdims=True)
    vector_sum = np.sum(vector_norm * np.exp(factor * 1.j * thetas), axis=1)
    index = np.abs(vector_sum)
    preference = np.angle(vector_sum) / factor
    return index, preference


def scatterplot(xs, ys, xlabel, ylabel, id_line=False, fontsize=15):
    """
    General scatterplot function
    :param xs:
    :param ys:
    :param xlabel:
    :param ylabel:
    :param id_line: boolean, whether or not to plot identity line
    :param fontsize:
    :return: figure handle
    """
    fig = plt.figure(figsize=(6, 6))
    if id_line:
        lmin = np.nanmin([np.nanquantile(xs, 0.01), np.nanquantile(ys, 0.01)])
        lmax = np.nanmax([np.nanquantile(xs, 0.99), np.nanquantile(ys, 0.99)])
        plt.plot([lmin, lmax], [lmin, lmax], '-', color='k')
    plt.scatter(xs, ys, marker='.', s=150, edgecolors=[1, 1, 1], alpha=1.0)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()
    return fig


def plot_cdfs(prefs, xlabel, fontsize=15):
    fig = plt.figure(figsize=(6, 6))
    for epoch in list(prefs.keys()):
        vals = prefs[epoch]
        plt.hist(
            vals[~np.isnan(vals)], bins=20, histtype='step', density=True, cumulative=True,
            linewidth=2, label=epoch)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel('Probability', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc='upper left')
    plt.show()
    return fig

















