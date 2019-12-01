"""
Creates summary metrics and plots for units in a recording session.

*** 3 Things to check before using this code ***

1) This module assumes that you are on the 'cert_master_fn' branch of the 'analysis' repo and that
the working directory can access the latest 'ibllib' - 'brainbox' branch, and the latest
'iblscripts' - 'certification' branch. If in doubt, in your OS terminal run:
    `pip install --upgrade git+https://github.com/int-brain-lab/ibllib.git@brainbox`
    `pip install --upgrade git+https://github.com/int-brain-lab/iblscripts.git@certification`

2) This module assumes that the required data for a particular eid is already saved in the
CACHE_DIR specified by `.one_params` (the default location to which ONE saves data when running the
`load` method). It is recommended to download *all* data for a particular eid:
    `from oneibl.one import ONE`
    `one = ONE()`
    # get eid
    `eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]`
    # download data
    one.load(eid, dataset_types=one.list(), clobber=False, download_only=True)

3) Ensure that you have the required, up-to-date versions of the following 3rd party package
dependencies in your environment: opencv-python, phylib. If in doubt, in your OS terminal run:
    `pip install opencv-python`
    `pip install --upgrade git+https://github.com/cortex-lab/phylib.git@master`

Here is a list of required data (alf objects) depending on the figures to be generated:
    a) required for any figure:
        clusters
        spikes
    b) if grating_response_summary or grating_response_ind:
        ephysData.raw
        _spikeglx_sync
        _iblrig_RFMapStim
        _iblrig_codeFiles
        _iblrig_taskSettings
    c) if using waveform metrics in unit_metrics_ind:
        ephysData.raw

When running this module as a script:
Run this as a script from within python:
`run path\to\plot`
or in a terminal, outside of python:
`python path\to\plot.py`

TODO combine summary figures into one figure
TODO add session info to figures
TODO metrics to add: 1) chebyshev's inequality, 2) cluster residuals, 3) silhouette, 4) d_prime,
    5) nn_hit_rate, 6) nn_miss_rate, 7) iso distance, 8) l_ratio
"""

import os
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from oneibl.one import ONE
import alf.io as aio
import brainbox as bb
from deploy.serverpc.certification import certification_pipeline
from v1_protocol import orientation
from v1_protocol import complete_raster_depth_per_spike
from v1_protocol import rf_mapping_old


def gen_figures(
    eid, probe='probe00', cluster_ids_summary=[], cluster_ids_selected=[], n_selected_cl=5,
    extract_stim_info=True, grating_response_summary=True, grating_response_selected=False,
    unit_metrics_summary=True, unit_metrics_selected=False,
    summary_metrics = ['rf', 'raster', 'feat_vars', 's', 'cv_fr', 'spks_missed', 'isi_viol',
                       'max_drift', 'cum_drift'],
    selected_metrics = ['s', 'cv_fr', 'spks_missed', 'isi_viol', 'max_drift', 'cum_drift'],
    grating_response_params={'pre_t': 0.5, 'post_t': 2.5, 'bin_t': 0.005, 'sigma': 0.025},
    auto_filt_cl_params={'min_amp': 100, 'min_fr': 0.5, 'max_fpr': 0.1, 'rp': 0.002},
    rf_params={'bin_sz': .05, 'lags': 4, 'method': 'corr'},
    save_dir=None):
    '''
    Generates figures for the V1 certification protocol for a given eid, probe, and clusters from a
    recording session.

    Parameters
    ----------
    eid : string
        The experiment ID for a recording session: the UUID of the session as per Alyx.
    probe : string (optional)
        The probe whose data will be used to generate the figures.
    cluster_ids_summary : array-like (optional)
        The clusters for which to generate `grating_response_summary` and/or `unit_metrics_summary`
        (if `[]`, clusters will be chosen via the filter parameters in `auto_filt_cl_params`,
        which is used in a call to `brainbox.processing.filter_units`)
    cluster_ids_selected : array-like (optional)
        The clusters for which to generate `grating_response_ind` and/or `unit_metrics_ind`.
        (if `[]`, up to `n_selected_cl` cluster ids will be selected from `cluster_ids_summary`)
    n_selected_cl : int
        The max number of `cluster_ids_selected` to choose if `cluster_ids_selected == []`.
    extract_stim_info : bool (optional)
        A flag for extracting stimulus info from the recording session into an alf directory.
    grating_response_summary : bool (optional)
        A flag for returning a figure with summary grating response plots for `cluster_ids_summary`
    grating_response_selected : bool (optional)
        A flag for returning a figure with single grating response plots for `cluster_ids_selected` 
    unit_metrics_summary : bool (optional)
        A flag for returning a figure with summary metrics plots for `cluster_ids_summary`.
    unit_metrics_selected : bool (optional)
        A flag for returning a figure with single unit metrics plots for `cluster_ids_selected`.
    summary_metrics : list (optional)
        The summary metrics plots to generate for the `unit_metrics_summary` figure. Possible
        values can include:
        'rf' :
        'raster' :
        'feat_vars' :
        's' : 
        'cv_fr' :
        'spks_missed' : 
        'isi_viol' :
        'max_drift' :
        'cum_drift' :
    selected_metrics : list (optional)
        The selected metrics plots to generate for the `unit_metrics_selected` figure. Possible
        values can include: 
        's' : 
        'cv_fr' :
        'spks_missed' : 
        'isi_viol' :
        'max_drift' :
        'cum_drift' :
    grating_response_params : dict (optional)
        Parameters for generating rasters based on time of grating stimulus presentation:
        'pre_t' : the time (s) shown before grating onset.
        'post_t' : the time (s) shown after grating onset.
        'bin_t' : the bin width (s) used to determine the number of spikes/bin 
        'sigma' : the width (s) of the smoothing kernel used to determine the number of spikes/bin
    auto_filt_cl_params : dict (optional)
        Parameters used in the call to `brainbox.processing.filter_units` for filtering clusters:
        'min_amp' : the minimum mean amplitude (in uV) of the spikes in the unit
        'min_fr' : the minimum firing rate (in Hz) of the unit
        'max_fpr' : the maximum false positive rate of the unit (using the fp formula in
                    Hill et al. (2011) J Neurosci 31: 8699-8705)
        'rp' : the refractory period (in s) of the unit. Used to calculate `max_fp`
    rf_params : dict (optional)
        Parameters used for the receptive field summary plot:
        'bin_sz' : the bin width (s) used
        'lags' : number of bins for calculating receptive field
        'method' : 'corr' or 'sta'
    save_dir : string (optional)
        The path to which to save generated figures. (if `None`, figures will not be automatically
        saved)

    Returns
    -------
    m: bunch
        A bunch containing metrics as fields.

    See Also
    --------
    deploy.serverpc.certification.certification_pipeline
    orientation
    complete_raster_depth_per_spike
    rf_mapping_old
    brainbox.metrics.metrics
    brainbox.plot.plot

    Examples
    --------
    1) For a given eid and probe in a particular recording session, generate grating response
    summary and unit metrics summary figures for the all units, and grating response selected and
    unit metrics selected figures for 5 randomly chosen units.
        # Add `ibllib`, `iblscripts`, and `analysis` repos to path *if necessary*:
        >>> import sys
        >>> import os
        >>> sys.path.extend(
                [os.path.abspath('.\\ibllib'), os.path.abspath('.\\iblscripts'),
                 os.path.abspath('.\\analysis')])
        # Get eid from ONE and load necessary dataset_types (this data should already be
        # downloaded to the local `CACHE_DIR` specified by ONE in `.one_params`):
        >>> from oneibl.one import ONE
        >>> one = ONE()
        >>> eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
        # Generate all V1 certification figures for the `eid` and `probe`
        >>> from v1_protocol import plot as v1_plot
        # *Note: 'probe_right' for this eid, new naming convention is 'probe00', 'probe01', etc.
        # `auto_filt_cl_params` here is relaxed so that all units are included.
        >>> m = v1_plot.gen_figures(eid, 'probe_right', 
                                    grating_response_selected=True, unit_metrics_selected=True,
                                    auto_filt_cl_params={'min_amp': 0, 'min_fr': 0,
                                                         'max_fpr': 100, 'rp': 0.002})
    
    2) For a given eid's 'probe_01' in a particular recording session, generate grating response
    summary and unit metrics summary figures (where the time shown before a grating is 1s, the time
    shown after a grating is 4s, the bin size used to compute the grating responses is 10 ms, and 
    the smoothing kernel used is 50 ms) for a filtered subset of units (where the minimum mean
    amplitude must be > 50 uV, the minimum firing rate must be > 2 Hz, and there is no upper limit
    to the estimated false positive ratio).
        # Add `ibllib`, `iblscripts`, and `analysis` repos to path *if necessary*:
        >>> import sys
        >>> import os
        >>> sys.path.extend(
                [os.path.abspath('.\\ibllib'), os.path.abspath('.\\iblscripts'),
                 os.path.abspath('.\\analysis')])
        # Get eid from ONE and load necessary dataset_types (this data should already be
        # downloaded to the local `CACHE_DIR` specified by ONE in `.one_params`):
        >>> from oneibl.one import ONE
        >>> one = ONE()
        >>> eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
        # Generate summary V1 certification figures for the `eid` and `probe` for filtered units:
        >>> from v1_protocol import plot as v1_plot
        # *Note: 'probe_right' for this eid, new naming convention is 'probe00', 'probe01', etc.
        >>> m = v1_plot.gen_figures(
                    eid, 'probe_right',
                    grating_response_summary=True, grating_response_selected=False,
                    unit_metrics_summary=True, unit_metrics_selected=False,
                    grating_response_params={'pre_t': 1, 'post_t': 4, 'bin_t': .01, 'sigma': .05},
                    auto_filt_cl_params={'min_amp': 50, 'min_fr': 2, 'max_fpr': 0, 'rp': .002})
    
    3) For a given eid's 'probe_01' in a particular recording session, generate only grating
    response selected and unit metrics selected figures based on the grating response parameters
    and unit filtering parameters in example 2), and save these figures to the working directory.
        # Add `ibllib`, `iblscripts`, and `analysis` repos to path *if necessary*:
        >>> import sys
        >>> import os
        >>> sys.path.extend(
                [os.path.abspath('.\\ibllib'), os.path.abspath('.\\iblscripts'),
                 os.path.abspath('.\\analysis')])
        # Get eid from ONE and load necessary dataset_types (this data should already be
        # downloaded to the local `CACHE_DIR` specified by ONE in `.one_params`):
        >>> from oneibl.one import ONE
        >>> one = ONE()
        >>> eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
        # Get filtered subset of units:
        >>> from pathlib import Path
        >>> import numpy as np
        >>> import alf.io as aio
        >>> import brainbox as bb
        >>> spks_path = one.load(eid, dataset_types='spikes.amps', clobber=False,
                                 download_only=True)[0]
        >>> probe_dir_part = np.where([part == 'probe_01' for part in Path(spks_path).parts])[0][0]
        >>> alf_probe_path = os.path.join(*Path(spks_path).parts[:probe_dir_part+1])
        >>> spks = aio.load_object(alf_probe_path, 'spikes')
        >>> filtered_units = \
                np.where(bb.processing.filter_units(spks, params={'min_amp': 50, 'min_fr': 2,
                                                                  'max_fpr': 0, 'rp': 0.002}))[0]
        # Generate selected V1 certification figures for the `eid` and `probe` for filtered units:
        >>> from v1_protocol import plot as v1_plot
        >>> save_dir = pwd
        # *Note: 'probe_right' for this eid, new naming convention is 'probe00', 'probe01', etc.
        >>> m = v1_plot.gen_figures(
                    eid, 'probe_right', cluster_ids_selected=filtered_units, auto_filt_cl=False,
                    grating_response_summary=False, grating_response_selected=True,
                    unit_metrics_summary=False, unit_metrics_selected=True,
                    grating_response_params={'pre_t': 1, 'post_t': 4, 'bin_t': 0.01, 'sigma': .05},
                    save_dir=save_dir)
    '''

    # Get necessary data via ONE:
    one = ONE()
    # Get important local paths from `eid`.
    spikes_path = one.load(eid, dataset_types='spikes.amps', clobber=False, download_only=True)[0]
    alf_dir_part = np.where([part == 'alf' for part in Path(spikes_path).parts])[0][0]
    session_path = os.path.join(*Path(spikes_path).parts[:alf_dir_part])
    alf_path = os.path.join(session_path, 'alf')
    alf_probe_path = os.path.join(alf_path, probe)
    ephys_file_path = os.path.join(session_path, 'raw_ephys_data', probe)
    # Ensure `alf_probe_path` exists:
    if not(os.path.isdir(alf_probe_path)):
        raise FileNotFoundError("The path to 'probe' ({}) does not exist! Check the 'probe' name."
                                .format(alf_probe_path))

    if extract_stim_info:  # get stimulus info and save in `alf_path`
        certification_pipeline.extract_stimulus_info_to_alf(session_path, save=True)
        # Copy `'_iblcertif'` files over to `alf_probe_path`
        for i in os.listdir(alf_path):
            if i[:10] == '_iblcertif':
                shutil.copy(os.path.join(alf_path, i), alf_probe_path)

    # Set `cluster_ids_summary` and `cluster_ids_selected`
    if len(cluster_ids_summary) == 0:  # filter all clusters according to `auto_filt_cl_params`
        print("'cluster_ids_summary' left empty, selecting filtered units.'", flush=True)
        spks = aio.load_object(alf_probe_path, 'spikes')
        cluster_ids_summary = \
            np.where(bb.processing.filter_units(spks, params=auto_filt_cl_params))[0]
        if cluster_ids_summary.size == 0:
            raise ValueError("'cluster_ids_summary' is empty! Check filtering parameters in\
                             'auto_filt_cl_params'.")
    if len(cluster_ids_selected) == 0:
        print("'cluster_ids_selected' left empty, selecting up to {} units from\
              'cluster_ids_summary'.".format(n_selected_cl), flush=True)
        if len(cluster_ids_summary) <= (n_selected_cl):  # select all of `cluster_ids_summary`
            cluster_ids_selected = cluster_ids_summary
        else:  # select up to 5 units from `cluster_ids_summary`
            cluster_ids_selected = np.random.choice(cluster_ids_summary,
                                                    size=n_selected_cl, replace=False)

    # Get visually responsive clusters and create appropriate figures
    if grating_response_summary or grating_response_selected:
        cluster_ids_summary_vr, cluster_ids_selected_vr = \
            orientation.get_vr_clusters(alf_probe_path, n_selected_cl)
    # Generate both summary & selected grating figures
        if grating_response_summary and grating_response_selected:
            orientation.plot_grating_figures(
                alf_probe_path, save_dir=save_dir, pre_time=grating_response_params['pre_t'],
                post_time=grating_response_params['post_t'],
                bin_size=grating_response_params['bin_t'],
                smoothing=grating_response_params['sigma'],
                cluster_ids_summary=cluster_ids_summary_vr,
                cluster_ids_selected=cluster_ids_selected_vr,
                n_rand_clusters=5)
        # Generate just summary grating figure
        elif grating_response_summary:
            orientation.plot_grating_figures(
                alf_probe_path, save_dir=save_dir, pre_time=grating_response_params['pre_t'],
                post_time=grating_response_params['post_t'],
                bin_size=grating_response_params['bin_t'],
                smoothing=grating_response_params['sigma'],
                cluster_ids_summary=cluster_ids_summary_vr,
                cluster_ids_selected=cluster_ids_selected_vr,
                n_rand_clusters=5, only_summary=True)
        # Generate just selected grating figure
        elif grating_response_selected:
            orientation.plot_grating_figures(
                alf_probe_path, save_dir=save_dir, pre_time=grating_response_params['pre_t'],
                post_time=grating_response_params['post_t'],
                bin_size=grating_response_params['bin_t'],
                smoothing=grating_response_params['sigma'],
                cluster_ids_summary=cluster_ids_summary_vr,
                cluster_ids_selected=cluster_ids_selected_vr,
                n_rand_clusters=5, only_selected=True)

    # Generate summary unit metrics figure
    if unit_metrics_summary:
        um_summary_plots(alf_probe_path, cluster_ids_summary, rf_params, ephys_file_path)
    
    # Generate selected unit metrics figure
    if unit_metrics_selected:
        um_selected_plots(alf_probe_path, cluster_ids_selected, ephys_file_path)


def um_summary_plots(alf_probe_path, clusters, rf_params):
    '''
    Computes/creates summary metrics and plots in a figure for all units in a recording session.

    Parameters
    ----------
    alf_probe_path : string
        The absolute path to an 'alf/probe' directory.
    clusters : array-like
        The clusters for which to generate the metrics summary plots.
    rf_params : dict
        Parameters used for the receptive field summary plot:
        'bin_sz' : the bin width (s) used
        'lags' : number of bins for calculating receptive field
        'method' : 'corr' or 'sta'

    Returns
    -------
    m: bunch
        A bunch containing metrics as fields. 

    See Also
    --------
    brainbox.metrics.metrics
    brainbox.plot.plot
    '''

    # rf histograms
    rf_mapping_old.histograms_rf_areas(alf_probe_path, clusters, rf_params)
    # raster
    complete_raster_depth_per_spike.scatter_with_boundary_times(alf_probe_path, clusters)
    # variances of amplitudes barplot
    spikes = aio.load_object(alf_probe_path, 'spikes')
    bb.plot.feat_vars(spikes)
    # waveform spatiotemporal correlation values bar plot
    
    # coefficient of variation of firing rates bar plot
    
    # % missing spikes bar plot


def um_selected_plots(alf_probe_path, clusters):
    '''
    Computes/creates metrics and plots in a figure for specified units in a recording session.

    Parameters
    ----------
    alf_probe_path : string
        The absolute path to an 'alf/probe' directory.
    clusters : array-like
        The clusters for which to generate the metrics selected unit plots.

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

    # Prompt user for eid and probe.
    
    # Generate grating response summary and unit metrics summary figures for "good units", and
    # grating response selected and unit metrics selected figures for the first 5 good units.
    print('end')
