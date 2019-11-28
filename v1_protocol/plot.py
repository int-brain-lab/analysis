"""
Creates summary metrics and plots for units in a recording session.

***3 Things to check before using this code***

1) This module assumes that your working directory can access the latest 'ibllib' - 'brainbox'
branch, and the latest 'iblscripts' - 'certification' branch. If in doubt, in your OS terminal run:
    `pip install --upgrade git+https://github.com/int-brain-lab/ibllib.git@brainbox`
    `pip install --upgrade git+https://github.com/int-brain-lab/iblscripts.git@iblscripts`

2) This module assumes that the required data for a particular eid is already saved in the
CACHE_DIR specified by `.one_params` (the default location to which ONE saves data when running the
 `load` method). It is recommended to download *all* data for a particular eid:
    `from oneibl.one import ONE`
    `one = ONE()`
    # get eid
    `eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]`
    # download data
    one.load(eid, dataset_types=one.list(), clobber=False, download_only=True)

3) Ensure you have the required, up-to-date versions of the following 3rd party package
dependencies in your environment: opencv-python, phylib. If in doubt, in your OS
terminal run:
    `pip install opencv-python`
    `pip install --upgrade git+https://github.com/cortex-lab/phylib.git@master`

Here is a list of required data (alf objects) depending on the figures to be generated:
required for any figure:
    clusters
    spikes
if grating_response_summary or grating_response_ind:
    ephysData.raw
    _spikeglx_sync
    _iblrig_RFMapStim
    _iblrig_codeFiles
    _iblrig_taskSettings
if using waveform metrics in unit_metrics_ind:
    ephysData.raw

When running this module as a script... 

Run this as a script from within python:
`run path\to\plot`
or in a terminal, outside of python:
`python path\to\plot.py`
"""

import sys
import os
# Add `ibllib`, `iblscripts`, and `analysis` repos to path.
sys.path.extend([os.path.abspath('.\\ibllib'), os.path.abspath('.\\iblscripts'),
                 os.path.abspath('.\\analysis')])
from pathlib import Path
import shutil
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import pandas as pd
from oneibl.one import ONE
import alf.io as aio
import brainbox as bb
from brainbox.processing import bincount2D
# Import code for extracting stimulus info to get v1 certification files (`_iblcertif_`).
import certification_pipeline
import orientation
from oneibl.one import ONE


def gen_figures(eid, probe='probe_00', cluster_ids=[], extract_stim_info=False,
                grating_response_summary=False, grating_response_selected=False,
                unit_metrics_summary=True, unit_metrics_selected=False,
                grating_response_params={'pre_t':0.5, 'post_t':2.5, 'bin_t':0.005, 'sigma':0.025},
                save_dir=None):
    '''
    Generates figures for the V1 certification protocol for a given eid and probe from a recording
    session.

    Parameters
    ----------
    eid : string
        The experiment ID for a recording session: the UUID of the session as per Alyx.
    probe : string
        The probe whose data will be used to generate the figures.
    cluster_ids : array-like
        The clusters for which to generate `grating_response_ind` and/or `unit_metrics_ind`.
    grating_response_summary : bool
        A flag for returning a figure with summary grating response plots based on all units.
    grating_response_selected : bool
        A flag for returning a figure with grating response plots for the selected units in 
        `cluster_ids`.
    unit_metrics_summary : bool
        A flag for returning a figure with summary metrics plots based on all units.
    unit_metrics_selected : bool
        A flag for returning a figure with single unit metrics plots for the selected units in
        `cluster_ids`.
    summary_metrics : list
        The summary metrics plots to generate for the `unit_metrics_summary` figure.
    selected_metrics : list
        The selected metrics plots to generate for the `unit_metrics_selected` figure.
    extract_stim_info : bool
        A flag for extracting stimulus info from the recording session into an alf directory.
    grating_response_params : dict
        Parameters for generating rasters based on time of grating stimulus presentation. 
        `pre_t` is the time (s) shown before grating onset.
        `post_t` is the time (s) shown after grating onset.
        `bin_t` is the bin width (s) used to determine the number of spikes/bin 
        `sigma` is the width (s) of the smoothing kernel used to determine the number of spikes/bin

    Returns
    -------
    m: bunch
        A bunch containing metrics as fields.

    See Also
    --------
    orientation
    deploy.serverpc.certification.certification_pipeline
    brainbox.metrics.metrics
    brainbox.plot.plot

    Examples
    --------
    1) Generate grating response summary and unit metrics summary figures for "good units"
    (returned by compute.good_units), and grating response selected and unit metrics selected
    for the first 5 good units.
    '''
    
    # Get necessary data via ONE:
    one = ONE()
    # Get important local paths from `eid`.
    spikes_path = one.load(eid, dataset_types='spikes.amps', clobber=False, download_only=True)[0]
    alf_dir_part = np.where([part == 'alf' for part in Path(spikes_path).parts])[0][0]
    session_path = os.path.join(*Path(spikes_path).parts[:alf_dir_part])
    alf_path = os.path.abspath(session_path + '\\alf')
    alf_probe_path = os.path.abspath(alf_path + '\\' + probe)

    if extract_stim_info:
        # Get stimulus info and save in `alf_path`
        certification_pipeline.extract_stimulus_info_to_alf(session_path, save=True)
        # Copy `'_iblcertif'` files over to `alf_probe_path`
        for i in os.listdir(alf_path):
            if i[:10] == '_iblcertif':
                shutil.copy(os.path.abspath(alf_path + '\\' + i), alf_probe_path)
    
    if grating_response_summary and grating_response_selected:
        orientation.plot_grating_figures(
            alf_probe_path, save_dir=save_dir, pre_time=grating_response_params['pre_t'],
            post_time=grating_response_params['post_t'], bin_size=grating_response_params['bin_t'],
            smoothing=grating_response_params['sigma'], cluster_idxs=cluster_ids,
            n_rand_clusters=5)
    elif grating_response_summary:
        orientation.plot_grating_figures(
            alf_probe_path, save_dir=save_dir, pre_time=grating_response_params['pre_t'],
            post_time=grating_response_params['post_t'], bin_size=grating_response_params['bin_t'],
            smoothing=grating_response_params['sigma'], cluster_idxs=cluster_ids,
            n_rand_clusters=5, only_summary=True)
    elif grating_response_selected:
        orientation.plot_grating_figures(
            alf_probe_path, save_dir=save_dir, pre_time=grating_response_params['pre_t'],
            post_time=grating_response_params['post_t'], bin_size=grating_response_params['bin_t'],
            smoothing=grating_response_params['sigma'], cluster_idxs=cluster_ids,
            n_rand_clusters=5, only_selected=True)
    
    if unit_metrics_summary:
        um_summary_plots(eid)
        
    if unit_metrics_selected:
        um_selected_plots()


import complete_raster_depth_per_spike
import rf_mapping_old

def um_summary_plots(eid):
    '''
    Computes/creates summary metrics and plots for all units in a given recording session.

    Parameters
    ----------
    eid : string
        The experiment ID, for a given recording session- the UUID of the session as per Alyx.

    Returns
    -------
    m: bunch
        A bunch containing metrics as fields. 

    See Also
    --------
    brainbox.metrics.metrics
    brainbox.plot.plot
    '''

    rf_mapping_old.histograms_rf_areas(eid)
    complete_raster_depth_per_spike.scatter_with_boundary_times(eid)
    one = ONE()
    D = one.load(eid[0], clobber=False, download_only=True)
    alf_path = Path(D.local_path[0]).parent
    spikes = aio.load_object(alf_path, 'spikes')
    bb.plot.feat_vars(spikes)




def um_selected_plots(clusters):
    '''
    Computes/creates metrics and plots for specified units in a given recording session.

    Parameters
    ----------
    eid : string
        The experiment ID, for a given recording session- the UUID of the session as per Alyx.

    Returns
    -------
    m: bunch
        A bunch containing metrics as fields. 

    See Also
    --------
    brainbox.metrics.metrics
    brainbox.plot.plot
    

    Examples
    --------
    1) Compute the similarity between the first and last 100 waveforms for unit1, across the 20
    channels around the channel of max amplitude.
        >>> import brainbox as bb
        >>> import alf.io as aio
        >>> import ibllib.ephys.spikes as e_spks
        # Get a spikes bunch, a clusters bunch, a units bunch, the channels around the max amp
        # channel for the unit, two sets of timestamps for the units, and the two corresponding
        # sets of waveforms for those two sets of timestamps. Then compute `s`.
        >>> e_spks.ks2_to_alf('path\\to\\ks_output', 'path\\to\\alf_output')
        >>> spks = aio.load_object('path\\to\\alf_output', 'spikes')
        >>> clstrs = aio.load_object('path\\to\\alf_output', 'clusters')
        >>> max_ch = max_ch = clstrs['channels'][1]
        >>> ch = np.arange(max_ch - 10, max_ch + 10)
        >>> units = bb.processing.get_units_bunch(spks)
        >>> ts1 = units['times']['1'][:100]
        >>> ts2 = units['times']['1'][-100:]
        >>> wf1 = bb.io.extract_waveforms('path\\to\\ephys_bin_file', ts1, ch)
        >>> wf2 = bb.io.extract_waveforms('path\\to\\ephys_bin_file', ts2, ch)
        >>> s = bb.metrics.wf_similarity(wf1, wf2)
    '''

def raster_complete(R, times, Clusters):
    '''
    Plot a rasterplot for the complete recording
    (might be slow, restrict R if so),
    ordered by insertion depth, with

    :param R: multidimensional binned neural activity
    :param times: ttl time stamps in sec
    :param Cluster: cluster ids
    '''

    T_BIN = 0.005
    plt.imshow(R, aspect='auto', cmap='binary', vmax=T_BIN / 0.001 / 4,
               origin='lower', extent=np.r_[times[[0, -1]], Clusters[[0, -1]]])

    plt.xlabel('Time (s)')
    plt.ylabel('Cluster #; ordered by depth')
    plt.show()

    # plt.savefig('/home/mic/Rasters/%s.png' %(trial_number))
    # plt.close('all')
    plt.tight_layout()

def plot_rf_distributions(rf_areas, plot_type='box'):
    """
    :param rf_areas:
    :param plot_type: 'box' | 'hist'
    :return: figure handle
    """

    # put results into dataframe for easier plotting
    results = []
    for sub_type, areas in rf_areas.items():
        for i, area in enumerate(areas):
            results.append(pd.DataFrame({
                'cluster_id': i,
                'area': area,
                'Subfield': sub_type.upper()}, index=[0]))
    results_pd = pd.concat(results, ignore_index=True)
    # leave out non-responsive clusters
    data_queried = results_pd[results_pd.area != 0]

    if plot_type == 'box':

        splt = sns.catplot(
            x='Subfield', y='area', kind='box', data=data_queried)
        splt.fig.axes[0].set_yscale('log')
        splt.fig.axes[0].set_ylabel('RF Area (pixels$^2$)')
        splt.fig.axes[0].set_ylim([1e-1, 1e4])

    elif plot_type == 'hist':
        bins = 10 ** (np.arange(-1, 4.25, 0.25))
        xmin = 1e-1
        xmax = 1e4
        ymin = 1e0
        ymax = 1e3
        splt = plt.figure(figsize=(12, 4))

        plt.subplot(121)
        plt.hist(data_queried[data_queried.Subfield ==
                              'ON']['area'], bins=bins, log=True)
        plt.xlabel('RF Area (pixels)')
        plt.xscale('log')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.ylabel('Cluster count')
        plt.title('ON Subfield')

        plt.subplot(122)
        plt.hist(data_queried[data_queried.Subfield ==
                              'OFF']['area'], bins=bins, log=True)
        plt.xlabel('RF Area (pixels)')
        plt.xscale('log')
        plt.xlim([xmin, xmax])
        plt.ylim([ymin, ymax])
        plt.title('OFF Subfield')

    plt.show()

    return splt


if __name__ == '__main__':

    # Prompt user for eid
    
    # Generate grating response summary and unit metrics summary figures for "good units", and
    # grating response selected and unit metrics selected figures for the first 5 good units.
    print('end')
