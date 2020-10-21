import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from oneibl.one import ONE
from ibllib.io import spikeglx
import alf.io
from scipy.io import savemat 
from brainbox.io.one import load_channel_locations
from scipy import signal
import brainbox as bb
from pathlib import Path
import pandas as pd
from numpy.random import randint
from brainbox.processing import bincount2D
import matplotlib.pyplot as plt
import ibllib.plots as iblplt
from collections import Counter
import os


plt.ion()

'''
This script includes functions to align lfp data, 
plot time seires overlayed to DLC-detected licks
'''

# MAXIMUM 3

# ['c9fec76e-7a20-4da4-93ad-04510a89473b',
#  'probe01',
#  ['hoferlab', 'Subjects', 'SWC_015', '2020-01-21', '002'],
#  3885],

#[['d33baf74-263c-4b37-a0d0-b79dcb80a764',
#  'probe00',
#  ['mainenlab', 'Subjects', 'ZM_2240', '2020-01-21', '001'],
#  1229],

# ['a8a8af78-16de-4841-ab07-fde4b5281a03',
#  'probe01',
#  ['angelakilab', 'Subjects', 'NYU-12', '2020-01-22', '001'],
#  448]


#eid_probe = 
#[['a8a8af78-16de-4841-ab07-fde4b5281a03','probe01'],
#['c9fec76e-7a20-4da4-93ad-04510a89473b','probe01'],
#['d33baf74-263c-4b37-a0d0-b79dcb80a764','probe00']]



def check_for_saturation(eid,probes):
    '''
    This functions reads in spikes for a given session,
    bins them into time bins and computes for how many of them,
    there is too little activity across all channels such that
    this must be an artefact (saturation)
    '''

    T_BIN = 0.2  # time bin in sec
    ACT_THR = 0.05  # maximal activity for saturated segment
    print('Bin size: %s [ms]' % T_BIN)
    print('Activity threshold: %s [fraction]' % ACT_THR)

    #probes = ['probe00', 'probe01']
    probeDict = {'probe00': 'probe_left', 'probe01': 'probe_right'}

    one = ONE()
    dataset_types = ['spikes.times', 'spikes.clusters']
    D = one.load(eid, dataset_types=dataset_types, dclass_output=True)
    alf_path = Path(D.local_path[0]).parent.parent
    print(alf_path)

    l = []
    for probe in probes:
        probe_path = alf_path / probe
        if not probe_path.exists():
            probe_path = alf_path / probeDict[probe]
            if not probe_path.exists():
                print("% s doesn't exist..." % probe)
                continue
        try:
            spikes = alf.io.load_object(probe_path, 'spikes')
        except:
            continue

        # bin spikes
        R, times, Clusters = bincount2D(
            spikes['times'], spikes['clusters'], T_BIN)

        saturated_bins = np.where(np.mean(R, axis=0) < 0.15)[0]


        if len(saturated_bins) > 1:
            print('WARNING: Saturation present!')
            print(probe)
            print('Number of saturated bins: %s of %s' %
                  (len(saturated_bins), len(times)))            
            
        l.append(['%s_%s' %(eid, probe), times[saturated_bins]])
        
    np.save('/home/mic/saturation_scan2/%s.npy' %eid, l) 
     
    return l 
    
def plot_saturation():
    
    '''
    plot the number of segments that are quiet in terms of spikes
    '''
        
    plt.ion()

    results_folder = '/home/mic/saturation_scan/'    
    t=list(os.walk(results_folder))[0][-1]
    
    sess_info = []
    sat_segs = []
    for ii in t:
        try:
            a = np.load(results_folder + ii)
        except:
            print("could't load %s" %ii)
            continue
        sess_info.append(a[:,0])
        sat_segs.append(a[:,1])        
        
    flat_sess_info = [item for sublist in sess_info for item in sublist]
    flat_sat_segs = [int(item) for sublist in sat_segs for item in sublist]

    maxes = np.where(np.array(flat_sat_segs)>10)    

    height = np.array(flat_sat_segs)[maxes] #flat_sat_segs
    bars = np.array(flat_sess_info)[maxes] 
    
    one = ONE()
    
    #flat_sess_info
    y_pos = np.arange(len(bars))
    
    # Create horizontal bars
    plt.barh(y_pos, height)
     
    # Create names on the y-axis
    plt.yticks(y_pos, bars, fontsize = 10)
    plt.xlabel('number of saturated 200 ms segments')
    plt.title('sessions with histology that meet behavior criterion for the BWM')

    sess_info = bars
    seg_info = height

    return seg_info, sess_info
    
    
def get_info(seg_info, sess_info):
 
    l = [] 
    one = ONE()

    for i in range(len(sess_info)):
        sess = sess_info[i]
        eid, probe = sess.split('_')
        D = one.load(eid, dataset_types=['trials.intervals'], dclass_output=True)   
        l.append([eid, probe, str(Path(D.local_path[0]).parent.parent).split('/')[5:],seg_info[i]]) 
        
    return l
 

def get_trials_from_times(eid):

    '''
    output number of quiet segments per trial number
    ''' 
  
    one = ONE()
    
    sat_times_path = '/home/mic/saturation_scan2/%s.npy' %eid
   
    sat_times_info = np.load(sat_times_path, allow_pickle=True)[0]
    sat_times = sat_times_info[1]
    
    D = one.load(eid, dataset_types=['trials.intervals'], dclass_output=True)
    alf_path = Path(D.local_path[0]).parent.parent / 'alf'
    trials = alf.io.load_object(alf_path, '_ibl_trials')

    trials_with_sat = []
    for t in range(len(trials['intervals'])):
        ter = trials['intervals'][t]
        for tt in sat_times:
            if ter[0] < tt < ter[1]:
                trials_with_sat.append(t)   
    
    C = Counter(trials_with_sat)    
    print(sat_times_info[0])
    print(len(C), 'of', len(trials['intervals']), 'trials have at least one saturation event')
    return C
    
    

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx     
    
def align_data(eid, one, trial_idx, probe):    
   
    # change .ap to .lf to get LFP instead of high frequency band
    # D['c607c5da-534e-4f30-97b3-b1d3e904e9fd']['probe01'] has Visp1, VISp2/3 and VISp4'
    # '3663d82b-f197-4e8b-b299-7b803a155b84', 'left', [8,8], lick example 

    lf_paths = one.load(eid, dataset_types=[ 'ephysData.raw.meta',
                                               'ephysData.raw.ch','ephysData.raw.sync',      
                                               'trials.intervals', 'trials.stimOn_times',
                                               'trials.feedbackType','trials.goCue_times',
                                               'trials.feedback_times','trials.contrastLeft',
                                               'trials.contrastRight'],
                                                download_only=True)    #'ephysData.raw.ap',
      
    lf_file = [x for x in lf_paths if probe in str(x) and 'ap.cbin' in str(x)][0]


    sr = spikeglx.Reader(lf_file)    
    
    sync_file = sr.file_bin.parent.joinpath(sr.file_bin.stem.replace('.ap', '.sync.npy'))
    sync = np.load(sync_file) 

    fs_sync = int(np.mean(np.diff(sync[:, 0]))) # sampled at 20 Hz?

    # upsample sync signal to sr
    sample2time = scipy.interpolate.interp1d(sync[:, 0] * sr.fs, sync[:, 1])
    
    alf_path = [x for x in lf_paths if 'alf' in str(x)][0].parent 
    trials = alf.io.load_object(alf_path, '_ibl_trials')
    
    # digitize to search idx only in small chunk
    times_to_align_to = trials['intervals'][:,0]
    binids = np.digitize(times_to_align_to, sync[:,1])


    # get lfp aligned for specific trial (trial_idx)       
    t = trials['intervals'][:,0][trial_idx]    
    times = sample2time(np.arange((binids[trial_idx]-1) * fs_sync * sr.fs, binids[trial_idx] * fs_sync * sr.fs))
    lfp_index = find_nearest(times, t)     
    startx = int(lfp_index + (binids[trial_idx]-1) * fs_sync * sr.fs) # in observations, 2500 Hz
    print(startx)
    
    t_end = trials['intervals'][:,1][trial_idx]         
    data_bounds = [int(startx), int((t_end - t) * sr.fs) + int(startx)] # in lfp frame idx
    print(data_bounds)
    data = sr[data_bounds[0]:data_bounds[1], :-1]
    times_data = sample2time(np.arange(data_bounds[0],data_bounds[1]))
    data = data - np.mean(data)

    return data, times_data


def get_ap_partial(eid, one, t_start, probe):

    # for a time point in seconds, get a 2 sec ap signal  
    #hofer sw 43: '7cdb71fb-928d-4eea-988f-0b655081f21c'
    
    seg_length = 3 # data segment length in seconds, starting at t_start     
    fs = 30000
    
    D=one.load(eid, dataset_types=['ephysData.raw.meta','ephysData.raw.sync'], dclass_output=True)
    meta_file = [x for x in D.local_path if probe in str(x) and 'ap' in str(x)][0]
    sync_file = meta_file.parent.joinpath(meta_file.stem.replace('.ap', '.sync.npy'))
    
    sync = np.load(sync_file)  
    fs_sync = int(np.mean(np.diff(sync[:, 0])))  
    # upsample sync signal to sr
    sample2time = scipy.interpolate.interp1d(sync[:, 0] * fs, sync[:, 1])    
    
    # digitize to search idx only in 20 sec chunk
    binids = np.digitize(sync[:, 0], sync[:,1])

    block_idx = np.where(sync[:, 0]>t_start)[0][0]

    # get ap aligned for specific t_start          
    times = sample2time(np.arange((binids[block_idx]-1) * fs_sync * fs, binids[block_idx] * fs_sync * fs))
    lfp_index = find_nearest(times, t_start)     
    startx = int(lfp_index + (binids[block_idx]-1) * fs_sync * fs) 
    #t_end = trials['intervals'][:,1][trial_idx]   
    t_end = t_start + seg_length # segment length in seconds     
    data_bounds = [int(startx), int((t_end - t_start) * fs) + int(startx)] # in lfp frame idx
   
    # the ap data is downloaded in 30000 frame chunks
    # make sure your start time is not at the limits of the recording
    start_chunk = data_bounds[0] // fs
    end_chunk = start_chunk + seg_length 
    

    pdict = {'probe00': 0,'probe01': 1}
    probe_idx = pdict[probe]
    dsets = one.alyx.rest(
        'datasets', 'list', session=eid,
        django='name__icontains,ap.cbin,collection__endswith,%s' %probe)
    for fr in dsets[probe_idx]['file_records']:
        if fr['data_url']:
            url_cbin = fr['data_url']

    dsets = one.alyx.rest(
        'datasets', 'list', session=eid,
        django='name__icontains,ap.ch,collection__endswith,%s' %probe)
    for fr in dsets[probe_idx]['file_records']:
        if fr['data_url']:
            url_ch = fr['data_url']
   
    ap_chunk = one.download_raw_partial(url_cbin, url_ch, start_chunk, end_chunk -1)
    
    print(url_cbin,url_ch) 
    
    times_data = sample2time(np.arange(start_chunk * fs, end_chunk * fs))
    
    return ap_chunk, times_data    

   
    
def plot_ap(data, times_data):
    plt.ion()
    obs, chans = data.shape 
    #chans = [5,200,380]
    #chans = 10
    fig, ax = plt.subplots()
    for i in range(chans)[::30]:
        tplot = data[:,range(chans)[i]] - np.mean(data[:,range(chans)[i]])
        plt.plot(times_data, tplot + i*2)    
 
    plt.title('ap_signal')
    plt.xlabel('time [s]')     
    ax.set_yticklabels([])
    plt.ylabel('every 15th channel')
 
def plot_all(eid,trial_idx, probe):
    one =ONE()
    
    ## ap signal
    #data, times_data = align_data(eid,one, trial_idx, probe)     
    #plot_ap(data, times_data)
      
    #plot_raster_single_trial(one, eid, trial_idx, probe)

    plot_rms(eid, probe)
    plot_power_spectrum_lfp(eid, probe)
    #return XYs
 
def plot_raster_single_trial(one, eid, trial_number, probe):
    '''
    Plot a rasterplot for a given trial,
    ordered by insertion depth, with
    'stimOn_times','feedback_times' and 'stimOff_times'
    '''
    
    dataset_types = ['clusters.depth','spikes.times', 'spikes.depths','spikes.clusters', 'trials.intervals']   
    
    D = one.load(eid, dataset_types = dataset_types, dclass_output=True)
    
    
    alf_path = Path(D.local_path[0]).parent.parent / 'alf'
    if  str(alf_path.parent)[-3:] == 'alf':
        alf_path = alf_path.parent
    probe_path = alf_path / probe

    spikes = alf.io.load_object(probe_path, 'spikes')
    trials = alf.io.load_object(alf_path, 'trials')   
    
    T_BIN = 0.01  # time bin in sec


    # bin spikes
    R, times, Clusters = bincount2D(
        spikes['times'], spikes['clusters'], T_BIN)

    # Order activity by cortical depth of neurons
    d = dict(zip(spikes['clusters'], spikes['depths']))
    y = sorted([[i, d[i]] for i in d])
    isort = np.argsort([x[1] for x in y])
    R = R[isort, :]

    # get trial number for each time bin
    trial_numbers = np.digitize(times, trials['intervals'][:,0])
    print('Range of trials: ', [trial_numbers[0], trial_numbers[-1]])
    
    plt.figure('2')   
    plt.title('%s_%s_trial: %s' %(eid, probe, trial_number))    
    trial_number = trial_number +1 
    a = list(trial_numbers)
    first = a.index(trial_number)
    last = len(a) - 1 - a[::-1].index(trial_number)

    plt.imshow(R[:, first:last], aspect='auto',
               cmap='binary', vmax=T_BIN / 0.001 / 4,
               extent=np.r_[times[[first, last]],
               Clusters[[0, -1]]], origin='lower')



    def restrict_timestamplist(q):

        li = []
        for i in q:
            if i > times[first] and i < times[last]:
                li.append(i)
        return li

    iblplt.vertical_lines(restrict_timestamplist(
        trials['stimOn_times']), ymin=0, ymax=Clusters[-1],
        color='m', linewidth=0.5, label='stimOn_times')

    iblplt.vertical_lines(restrict_timestamplist(
        trials['feedback_times']), ymin=0, ymax=Clusters[-1],
        color='b', linewidth=0.5, label='feedback_times')

#    iblplt.vertical_lines(restrict_timestamplist(
#        trials['stimOff_times']), ymin=0, ymax=Clusters[-1],
#        color='g', linewidth=0.5, label='stimOff_times')

    plt.xlabel('Time (s)')
    plt.ylabel('Cluster #; ordered by depth')
    plt.legend()
    plt.tight_layout()
    plt.show() 


def plot_rms(eid, probe_label):

    # https://int-brain-lab.github.io/iblenv/notebooks_external/docs_get_rms_data.html
    
    plt.ion() 
    
    # instantiate ONE
    one = ONE()

    # Specify subject, date and probe we are interested in
#    subject = 'CSHL049'
#    date = '2020-01-08'
#    sess_no = 1
#    probe_label = 'probe00'
#    eid = one.search(subject=subject, date=date, number=sess_no)[0]

    # Specify the dataset types of interest
    dtypes = ['_iblqc_ephysTimeRms.rms',
              '_iblqc_ephysTimeRms.timestamps',
              'channels.rawInd',
              'channels.localCoordinates']

    # Download the data and get paths to downloaded data
    _ = one.load(eid, dataset_types=dtypes, download_only=True)
    ephys_path = one.path_from_eid(eid).joinpath('raw_ephys_data', probe_label)
    alf_path = one.path_from_eid(eid).joinpath('alf', probe_label)

    session_name = '_'.join(str(ephys_path).split('/')[5:10])
    # Index of good recording channels along probe
    chn_inds = np.load(alf_path.joinpath('channels.rawInd.npy'))
    # Position of each recording channel along probe
    chn_pos = np.load(alf_path.joinpath('channels.localCoordinates.npy'))
    # Get range for y-axis
    depth_range = [np.min(chn_pos[:, 1]), np.max(chn_pos[:, 1])]

    # RMS data associated with AP band of data
    rms_ap = alf.io.load_object(ephys_path, 'ephysTimeRmsAP', namespace='iblqc')
    rms_ap_data = 20* np.log10(rms_ap['rms'][:, chn_inds] * 1e6)  # convert to uV

#    # Median subtract to clean up the data
#    median = np.mean(np.apply_along_axis(lambda x: np.median(x), 1, rms_ap_data))
#    # Add back the median so that the actual values in uV remain correct
#    rms_ap_data_median = np.apply_along_axis(lambda x: x - np.median(x), 1, rms_ap_data) + median

    # Get levels for colour bar and x-axis
    ap_levels = np.quantile(rms_ap_data, [0.1, 0.9])
    ap_time_range = [rms_ap['timestamps'][0], rms_ap['timestamps'][-1]]

    # RMS data associated with LFP band of data
    rms_lf = alf.io.load_object(ephys_path, 'ephysTimeRmsLF', namespace='iblqc')
    rms_lf_data = rms_lf['rms'][:, chn_inds] * 1e6  # convert to uV
    # Median subtract to clean up the data
#    median = np.mean(np.apply_along_axis(lambda x: np.median(x), 1, rms_lf_data))
#    rms_lf_data_median = np.apply_along_axis(lambda x: x - np.median(x), 1, rms_lf_data) + median

    lf_levels = np.quantile(rms_lf_data, [0.1, 0.9])
    lf_time_range = [rms_lf['timestamps'][0], rms_lf['timestamps'][-1]]

    # Create figure
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    # Plot the AP rms data
    ax0 = ax[0]
#    rms_ap_plot = ax0.imshow(rms_ap_data.T, extent=np.r_[ap_time_range, depth_range],
#                             cmap='plasma', vmin=ap_levels[0], vmax=ap_levels[1], origin='lower')
    rms_ap_plot = ax0.imshow(rms_ap_data.T, extent=np.r_[ap_time_range, depth_range],
                             cmap='plasma', vmin=0, vmax=100, origin='lower')                             
                             
    cbar_ap = fig.colorbar(rms_ap_plot, ax=ax0)
    cbar_ap.set_label('AP RMS (uV)')
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Depth along probe (um)')
    ax0.set_title('RMS of AP band')

    # Plot the LFP rms data
    ax1 = ax[1]
#    rms_lf_plot = ax1.imshow(rms_lf_data.T, extent=np.r_[lf_time_range, depth_range],
#                             cmap='inferno', vmin=lf_levels[0], vmax=lf_levels[1], origin='lower')

    rms_lf_plot = ax1.imshow(rms_lf_data.T, extent=np.r_[lf_time_range, depth_range],
                             cmap='inferno', vmin=0, vmax=1500, origin='lower')
    cbar_lf = fig.colorbar(rms_lf_plot, ax=ax1)
    cbar_lf.set_label('LFP RMS (uV)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Depth along probe (um)')
    ax1.set_title('RMS of LFP band')
    
    plt.suptitle('%s_%s \n %s' %(eid, probe_label, session_name))
    plt.savefig('/home/mic/saturation_analysis/rms_plots/%s_%s.png' %(eid, probe_label))
    plt.show()
    
    
def plot_power_spectrum_lfp(eid, probe_label):
    # instantiate ONE
    one = ONE()

#    # Specify subject, date and probe we are interested in
#    subject = 'CSHL049'
#    date = '2020-01-08'
#    sess_no = 1
#    probe_label = 'probe00'
#    eid = one.search(subject=subject, date=date, number=sess_no)[0]

    # Specify the dataset types of interest
    dtypes = ['_iblqc_ephysSpectralDensity.freqs',
              '_iblqc_ephysSpectralDensity.power',
              'channels.rawInd',
              'channels.localCoordinates']

    # Download the data and get paths to downloaded data
    _ = one.load(eid, dataset_types=dtypes, download_only=True)
    ephys_path = one.path_from_eid(eid).joinpath('raw_ephys_data', probe_label)
    alf_path = one.path_from_eid(eid).joinpath('alf', probe_label)

    # Index of good recording channels along probe
    chn_inds = np.load(alf_path.joinpath('channels.rawInd.npy'))
    # Position of each recording channel along probe
    chn_pos = np.load(alf_path.joinpath('channels.localCoordinates.npy'))
    # Get range for y-axis
    depth_range = [np.min(chn_pos[:, 1]), np.max(chn_pos[:, 1])]

    # Load in power spectrum data
    lfp_spectrum = alf.io.load_object(ephys_path, 'ephysSpectralDensityLF', namespace='iblqc')
    lfp_freq = lfp_spectrum['freqs']
    lfp_power = lfp_spectrum['power'][:, chn_inds]

    # Define a frequency range of interest
    freq_range = [0, 300]
    freq_idx = np.where((lfp_freq >= freq_range[0]) &
                        (lfp_freq < freq_range[1]))[0]

    # Limit data to freq range of interest and also convert to dB
    lfp_spectrum_data = 10 * np.log(lfp_power[freq_idx, :])
    dB_levels = np.quantile(lfp_spectrum_data, [0.1, 0.9])

    # Create figure
    fig, ax = plt.subplots()
    # Plot the LFP spectral data
    spectrum_plot = ax.imshow(lfp_spectrum_data.T, extent=np.r_[freq_range, depth_range],
                              cmap='viridis', vmin=dB_levels[0], vmax=dB_levels[1], origin='lower',
                              aspect='auto')
    cbar = fig.colorbar(spectrum_plot, ax=ax)
    cbar.set_label('LFP power (dB)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Depth along probe (um)')
    #ax.set_title('Power Spectrum of LFP')

#    plt.show()

    session_name = '_'.join(str(ephys_path).split('/')[5:10])
    plt.suptitle('%s_%s \n %s' %(eid, probe_label, session_name))
    plt.savefig('/home/mic/saturation_analysis/PSD_plots/%s_%s.png' %(eid, probe_label))

