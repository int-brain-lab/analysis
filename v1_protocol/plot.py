"""
Creates summary metrics and plots for units in a recording session.

*** Things to check before using this code ***

1) This module assumes that the required data for a particular eid is already saved in the
CACHE_DIR specified by `.one_params` (the default location to which ONE saves data when running the
`load` method). Although the default call to the master plotting function (`gen_figures`) will not
create figures from raw data, it is still recommended to download *all* data for a particular eid:
    >>> from oneibl.one import ONE
    >>> one = ONE()
    # get eid
    >>> eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
    # download data
    >>> one.load(eid, dataset_types=one.list(), clobber=False, download_only=True)
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

For additional details on and examples of running `gen_figures`, see the 
`using_master_plotting_function` script in this same directory.

TODO metrics to add: 1) chebyshev's inequality, 2) cluster residuals, 3) silhouette, 4) d_prime,
    5) nn_hit_rate, 6) nn_miss_rate, 7) iso distance, 8) l_ratio
"""

import os
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from oneibl.one import ONE
import alf.io as aio
import brainbox as bb
from ibllib.io import certification_protocol
from v1_protocol import orientation
from v1_protocol import complete_raster_depth_per_spike as raster_depth
from v1_protocol import rf_mapping


def gen_figures(
    eid, probe='probe00', cluster_ids_summary=None, cluster_ids_selected=None, n_selected_cl=4,
    extract_stim_info=True, grating_response_summary=True, grating_response_selected=False,
    unit_metrics_summary=True, unit_metrics_selected=False,
    summary_metrics = ['feat_vars', 'spks_missed', 'isi_viol', 'max_drift', 'cum_drift'],
    selected_metrics = ['isi_viol', 'spks_missed', 'cv_fr'],
    filt_params={'min_amp': 100, 'min_fr': 0.5, 'max_fpr': 0.1, 'rp': 0.002},
    grating_response_params={'pre_t': 0.5, 'post_t': 2.5, 'bin_t': 0.005, 'sigma': 0.025},
    summary_metrics_params={'bins': 'auto', 'rp': 0.002, 'spks_per_bin': 20, 'sigma': 5,
                            'n_ch': 10, 'fr_hist_win': 0.01, 'fr_ma_win': 0.5, 'n_cv_bins': 10,
                            'n_ch_probe': 385},
    selected_metrics_params={'spks_per_bin': 20, 'sigma': 5, 'rp': 0.002, 'bins': 'auto',
                             'n_ch': 10, 'fr_hist_win': 0.01, 'fr_ma_win': 0.5, 'n_cv_bins': 10,
                             'n_ch_probe': 385, 'isi_win': 0.01},
    rf_params={'method': 'corr', 'binsize': 0.025, 'lags': 8, 'n_depths': 30, 'use_svd': False},
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
        (if `[]`, clusters will be chosen via the filter parameters in `filt_params`,
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
        The summary metrics plots to generate for the `unit_metrics_summary` figures for
        `cluster_ids_summary`. Possible values can include:
            'feat_vars' : Bar plot of variances of empirical amplitude distribution.
            's' : Hist of spatiotemporal waveform correlation metric.
            'cv_fr' : Hist of coefficient of variation of firing rate.
            'spks_missed' : Hist of fraction of spikes missed.
            'isi_viol' : Hist of isi violations.
            'max_drift' : Hist of max drift metric.
            'cum_drift' : Hist of cumulative drift metric.
    selected_metrics : list (optional)
        The selected metrics plots to generate for the `unit_metrics_selected` figure for
        `cluster_ids_summary`. Possible values can include: 
            'isi_viol' : Plot of the histogram of isi violations.
            'spks_missed' : Plot of the pdf of the spike amplitude distribution.
            'cv_fr' : Plot of the firing rate.
            'amp_heatmap' : Plot of the amplitude heatmap.
            'peth' : Peri-event time histogram.
            's' : Plots of waveforms across `'n_ch'`.
    grating_response_params : dict (optional)
        Parameters for generating rasters based on time of grating stimulus presentation:
            'pre_t' : float
                The time (s) shown before grating onset.
            'post_t' : float
                The time (s) shown after grating onset.
            'bin_t' : float
                The bin width (s) used to determine the number of spikes/bin.
            'sigma' : float
                The width (s) of the smoothing kernel used to determine the number of spikes/bin.
    filt_params : dict (optional)
        Parameters used in the call to `brainbox.processing.filter_units` for filtering clusters:
            'min_amp' : float
                The minimum mean amplitude (in uV) of the spikes in the unit.
            'min_fr' : float
                The minimum firing rate (in Hz) of the unit.
            'max_fpr' : float
                The maximum false positive rate of the unit (using the fp formula in Hill et al.
                (2011) J Neurosci 31: 8699-8705)
            'rp' : float 
                The refractory period (in s) of the unit. Used to calculate `max_fp`.
    summary_metrics_params : dict
        Parameters used for the summary metrics figure:
            'bins' : int OR sequence OR string. 
                The number of bins (or the bins, or the method used to compute the bins) used for
                computing the histograms. (see `numpy.histogram_bin_edges`).
            'rp' : float 
                The refractory period (in s) of the unit
            'spks_per_bin' : int 
                The number of spikes per bin from which to compute the spike feature histogram for
                `spks_missed`.
            'sigma' : float
                The standard deviation for the gaussian kernel used to compute the pdf from the
                spike feature histogram for `spks_missed`.
            'n_ch' : int 
                The number of channels used to compute `s`.
            'fr_hist_win' : float
                The time window (in s) to use for computing spike counts for the instantaneous
                firing rate.
            'fr_ma_win' : float
                The time window (in s) to use as a moving slider to compute the instantaneous
                firing rate.
            'n_cv_bins' : int
                The number of equally spaced bins in time in which to compute the coefficient of
                variation of the firing rate.
            'n_ch_probe' : int
                The total number of channels from the recording (The number of rows in the binary
                ephys file).
            'isi_win' : float
                The x-axis (i.e. time (in s)) used for plotting the individual unit isi histograms
    selected_metrics_params : dict
        Parameters used for the selected metrics figure. Includes all values in
        `summary_metrics_params`, plus:
            'isi_win' : float
                The x-axis (i.e. time (in s)) used for plotting the individual unit isi histograms
    rf_params : dict (optional)
        Parameters used for the receptive field summary plot:
            'method' : string
                The method used to compute receptive fields ('corr' or 'sta').
            'bin_sz' : float 
                The bin width (s) used
            'lags' : int 
                The number of bins for calculating receptive field.
            'n_depths' : int 
                The number of depths to aggregate clusters over.
            'use_svd' : bool
                `True` plots 1st spatial SVD component of rf; `False` plots time lag with
                peak response.
    save_dir : string (optional)
        The path to which to save generated figures. (if `None`, figures will not be automatically
        saved)

    Returns
    -------
    m : bunch
        A bunch containing metrics as fields. Possible keys:
            'osi' : dict
                The orientation selectivity index of units in
                `cluster_sets['cluster_ids_summary_vr]'` at the beginning and end of session.
                Possible keys:
                    'beg'
                    'end'
            'orientation_pref' : dict
                The orientation preference of units in `cluster_sets['cluster_ids_summary_vr]'` at
                the begininng and end of session. Possible keys:
                    'beg'
                    'end'
            'frac_resp_by_depth' : dict
                The fraction of units in `cluster_sets['cluster_ids_summary_vr]'` responsive by 
                depth at beginning or end of session. Possible keys:
                    'fraction' : dict
                        The fraction of units responsive at the beginning and end of a session, at
                        the given depths in `m['frac_resp_by_depth']['depth']`. Possible keys:
                            'beg' : ndarray
                                The fraction responsive at each depth at session beginning.
                            'end' : ndarray
                                The fraction responsive at each depth at session end.
                    'depth' : ndarray
                        The depths used to compute the fraction responsive by depth.
            'var_amps' : ndarray
                The variance of the amplitude distribution for each unit in
                `cluster_sets['cluster_ids_summary']`.
            'fraction_missing' : ndarray 
                Estimated fraction missing spikes for each unit in
                `cluster_sets['cluster_ids_summary']`.
            'isi_viol' : ndarray 
                Fraction of isi violations for each unit in `cluster_sets['cluster_ids_summary']`.
            'max_drift' : ndarray
                Max drift values for each unit in `cluster_sets['cluster_ids_summary']`.
            'cum_drift' : ndarray
                Cumulative drift values for each unit in `cluster_sets['cluster_ids_summary']`.
    cluster_sets : dict
        Contains the ids of different sets of clusters used to generate the different figures.
        Possible keys:
            'cluster_ids_summary'
            'cluster_ids_selected'
            'cluster_ids_summary_vr' : a visually responsive subset of 'cluster_ids_summary'.
            'cluster_ids_selected_vr' : a subset of `n_selected_cl` clusters from
                'cluster_ids_summary_vr'. 
    fig_h : dict
        Contains the handles to the figures generated. Possible keys:
            'fig_gr_summary' : the grating responses summary figure
            'fig_gr_selected' : the grating responses selected units figure
            'fig_um_summary' : the unit metrics summary figure
            'fig_um_selected' : the unit metrics selected figure

    See Also
    --------
    ibllib.io.certification_protocol
    orientation
    complete_raster_depth_per_spike
    rf_mapping_old
    brainbox.metrics.metrics
    brainbox.plot.plot

    Examples
    --------
    1) For a given eid and probe in a particular recording session, generate grating response
    summary and unit metrics summary figures for *all* units, and grating response selected and
    unit metrics selected figures for 4 randomly chosen units.
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
        # `filt_params` here is relaxed so that all units are included.
        >>> m = v1_plot.gen_figures(
                    eid, 'probe_right', n_selected_cl=4, grating_response_selected=True,
                    unit_metrics_selected=True, 
                    filt_params={'min_amp': 0, 'min_fr': 0, 'max_fpr': 100, 'rp': 0.002})
    
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
                    filt_params={'min_amp': 50, 'min_fr': 2, 'max_fpr': 0, 'rp': .002})
    
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
        >>> filtered_units = bb.processing.filter_units(
                spks, min_amp=50, min_fr=2, max_fpr=0, rp=0.002})[0]
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

    # Initialize outputs #
    #--------------------#
    m = bb.core.Bunch()
    cluster_sets = {}
    fig_h = {}

    # Get necessary data via ONE #
    #----------------------------#
    one = ONE()
    # Get important local paths from `eid`.
    spikes_path = one.load(eid, dataset_types='spikes.amps', clobber=False, download_only=True)[0]
    alf_dir_part = np.where([part == 'alf' for part in Path(spikes_path).parts])[0][0]
    session_path = os.path.join(*Path(spikes_path).parts[:alf_dir_part])
    alf_path = os.path.join(session_path, 'alf')
    alf_probe_path = os.path.join(alf_path, probe)
    # Ensure `alf_probe_path` exists and `ephys_file_path` exists:
    if not(os.path.isdir(alf_probe_path)):
        raise FileNotFoundError("The path to 'probe' ({}) does not exist! Check the 'probe' name."
                                .format(alf_probe_path))
    ephys_file_dir = os.path.join(session_path, 'raw_ephys_data', probe)
    # Get `ap` ephys file.
    for file in os.listdir(ephys_file_dir):
        if 'ap' in file and 'bin' in file:
            ephys_file_path = os.path.join(ephys_file_dir, file)
    if extract_stim_info:  # get stimulus info and save in `alf_path`
        certification_protocol.extract_stimulus_info_to_alf(session_path, save=True)
        # Copy `'_iblcertif'` files over to `alf_probe_path`
        for i in os.listdir(alf_path):
            if i[:10] == '_iblcertif':
                shutil.copy(os.path.join(alf_path, i), alf_probe_path)
    # Get units bunch.
    spks_b = aio.load_object(alf_probe_path, 'spikes')
    print('Re-formatting alf data to save time during plotting. May take a few minutes...', 
          flush=True, end='')
    units_b = bb.processing.get_units_bunch(spks_b)
    print('done')
    
    # Set `cluster_ids_summary` and `cluster_ids_selected` #
    #------------------------------------------------------#
    if cluster_ids_summary is None:  # filter all clusters according to `filt_params`
        print("'cluster_ids_summary' left empty, selecting filtered units.")
        T = spks_b['times'][-1] - spks_b['times'][0]
        cluster_ids_summary = bb.processing.filter_units(
            units_b, T, min_amp=filt_params['min_amp'], min_fr=filt_params['min_fr'],
            max_fpr=filt_params['max_fpr'], rp=filt_params['rp'])
        if cluster_ids_summary.size == 0:
            raise ValueError("'cluster_ids_summary' is empty! Check filtering parameters in\
                             'filt_params'.")
    if cluster_ids_selected is None:
        print("'cluster_ids_selected' left empty, selecting up to {} units from\
              'cluster_ids_summary'.".format(n_selected_cl))
        if len(cluster_ids_summary) <= (n_selected_cl):  # select all of `cluster_ids_summary`
            cluster_ids_selected = cluster_ids_summary
        else:  # select up to 5 units from `cluster_ids_summary`
            cluster_ids_selected = np.random.choice(cluster_ids_summary,
                                                    size=n_selected_cl, replace=False)
    cluster_sets['cluster_ids_summary'] = cluster_ids_summary
    cluster_sets['cluster_ids_selected'] = cluster_ids_selected
    fig_list_name = []  # print this list at end of function to show which figures were generated

    # Get visually responsive clusters and generate grating response figures #
    #------------------------------------------------------------------------#
    if grating_response_summary or grating_response_selected:
        print('Generating grating response figure(s)...', flush=True, end='')
        # Get visually responsive clusters as subset of `cluster_ids_summary`
        cluster_ids_summary_vr, cluster_ids_selected_vr = \
            orientation.get_vr_clusters(alf_probe_path, clusters=cluster_ids_summary,
                                        n_selected_cl=n_selected_cl)
        cluster_sets['cluster_ids_summary_vr'] = cluster_ids_summary_vr
        cluster_sets['cluster_ids_selected_vr'] = cluster_ids_selected_vr
        # Generate grating figure(s)
        grating_figs, grating_metrics = orientation.plot_grating_figures(
            alf_probe_path, save_dir=save_dir, pre_time=grating_response_params['pre_t'],
            post_time=grating_response_params['post_t'],
            bin_size=grating_response_params['bin_t'],
            smoothing=grating_response_params['sigma'],
            cluster_ids_summary=cluster_ids_summary_vr,
            cluster_ids_selected=cluster_ids_selected_vr,
            n_rand_clusters=n_selected_cl,
            plot_summary=grating_response_summary,
            plot_selected=grating_response_selected)
        fig_h.update(grating_figs) 
        m.update(grating_metrics)
        fig_list_name.extend(['grating_response_summary', 'grating_response_selected']) 
        print('done')

    # Generate summary unit metrics figure #
    #--------------------------------------#
    if unit_metrics_summary:
        print('Generating summary metrics figure...', flush=True, end='')
        fig_um_summary, m = um_summary_plots(
            cluster_ids_summary, summary_metrics, units_b, alf_probe_path, ephys_file_path, m,
            summary_metrics_params, rf_params, save_dir=save_dir)
        fig_h['fig_um_summary'] = fig_um_summary
        fig_list_name.extend(['unit_metrics_summary'])
        print('done')
    
    # Generate selected unit metrics figure #
    #---------------------------------------#
    if unit_metrics_selected:
        print('Generating selected units metrics figure...', flush=True, end='')
        fig_um_selected, m = um_selected_plots(
            cluster_ids_selected, selected_metrics, units_b, alf_probe_path, ephys_file_path, m,
            selected_metrics_params, save_dir=save_dir)
        fig_h['fig_um_selected'] = fig_um_selected
        fig_list_name.extend(['unit_metrics_selected'])
        print('done')
    
    print('\n\nFinished generating figures {} for session {}'.format(fig_list_name, session_path))
    
    return m, cluster_sets, fig_h

def um_summary_plots(clusters, metrics, units_b, alf_probe_path, ephys_file_path, m,
                     metrics_params, rf_params, save_dir=None):
    '''
    Computes/creates summary metrics and plots in a figure for all units in a recording session.

    Parameters
    ----------
    clusters : list
        The clusters for which to generate the metrics summary plots.
    metrics : list
        The summary metrics plots to generate for the `unit_metrics_summary` figure. Possible
        values can include:
        'feat_vars' :
        's' : 
        'cv_fr' :
        'spks_missed' : 
        'isi_viol' :
        'max_drift' :
        'cum_drift' :
    units_b : bunch
        A units bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all units.
    alf_probe_path : string
        The absolute path to an 'alf/probe' directory.
    ephys_file_path : string
        The path to the binary ephys file.
    metrics_params : dict
        Parameters used for the summary metrics figure:
            'bins' : int OR sequence OR string. 
                The number of bins (or the bins, or the method used to compute the bins) used for
                computing the histograms. (see `numpy.histogram_bin_edges`).
            'rp' : float 
                The refractory period (in s) of the unit
            'spks_per_bin' : int 
                The number of spikes per bin from which to compute the spike feature histogram for
                `spks_missed`.
            'sigma' : float
                The standard deviation for the gaussian kernel used to compute the pdf from the
                spike feature histogram for `spks_missed`.
            'n_ch' : int 
                The number of channels used to compute `s`.
            'fr_hist_win' : float
                The time window (in s) to use for computing spike counts for the instantaneous
                firing rate.
            'fr_ma_win' : float
                The time window (in s) to use as a moving slider to compute the instantaneous
                firing rate.
            'n_cv_bins' : int
                The number of equally spaced bins in time in which to compute the coefficient of
                variation of the firing rate.
            'n_ch_probe' : int
                The total number of channels from the recording (The number of rows in the binary
                ephys file).
    rf_params : dict
        Parameters used for the receptive field summary plot:
        'bin_sz' : the bin width (s) used
        'lags' : number of bins for calculating receptive field
        'method' : 'corr' or 'sta'
    m : bunch
        A bunch containing metrics as fields.
    save_dir : string
        The path to which to save generated figures. (if `None`, figures will not be automatically
        saved)

    Returns
    -------
    m : bunch
        A bunch containing metrics as fields.
    fig : figure
        A handle to the figure generated.

    See Also
    --------
    brainbox.metrics.metrics
    brainbox.plot.plot
    
    Examples
    --------
    '''

    # Extract parameter values #
    #--------------------------#
    bins = metrics_params['bins']
    rp = metrics_params['rp']
    spks_per_bin = metrics_params['spks_per_bin']
    sigma = metrics_params['sigma']
    n_ch = metrics_params['n_ch']
    fr_hist_win = metrics_params['fr_hist_win']
    fr_ma_win = metrics_params['fr_ma_win']
    n_cv_bins = metrics_params['n_cv_bins']
    n_ch_probe = metrics_params['n_ch_probe']
    rf_method = rf_params['method']
    rf_binsize = rf_params['binsize']
    rf_lags = rf_params['lags']
    rf_n_depths = rf_params['n_depths']
    use_svd = rf_params['use_svd']
    
    # Set figure #
    #------------#
    ncols = 5  # axes per row of figure
    nrows = np.int(np.ceil(len(metrics) / ncols)) + 1
    fig = plt.figure(figsize=[16,8])
    fig.set_tight_layout(False)
    fig.suptitle('Summary Metrics')
    n_cur_ax = ncols + 1

    # Always output raster as half of first row 
    # TODO change this so that raster takes up ~60% of first row
    raster_ax = fig.add_subplot(nrows, 2, 1)
    raster_depth.scatter_with_boundary_times(alf_probe_path, clusters, ax=raster_ax)  # raster
    # Always output rf maps as second half of first row
    rf_map_ax = [fig.add_subplot(nrows, 4, 3), fig.add_subplot(nrows, 4, 4)]
    rf_mapping.plot_rfs_by_depth_wrapper(  # rf maps
        alf_probe_path, axes=rf_map_ax, cluster_ids=clusters, method=rf_method, binsize=rf_binsize,
        lags=rf_lags, n_depths=rf_n_depths, use_svd=use_svd)  
    
    # Get alf objects for this session (needed for some metrics calculations below)
    clstrs_b = aio.load_object(alf_probe_path, 'clusters')
    
    if 'feat_vars' in metrics:  # variances of amplitudes barplot
        feat_vars_ax = fig.add_subplot(nrows, ncols, n_cur_ax)
        var_amps, _ = bb.plot.feat_vars(units_b, units=clusters, feat_name='amps', ax=feat_vars_ax)
        m['var_amps'] = var_amps
        n_cur_ax += 1
    if 's' in metrics:  # waveform spatiotemporal correlation values hist
        s_ax = fig.add_subplot(nrows, ncols, n_cur_ax)
        s = s_hist(ephys_file_path, units_b, clstrs_b, units=clusters, bins=bins, n_ch=n_ch,
                   n_ch_probe=n_ch_probe, ax=s_ax)
        m['s'] = s
        n_cur_ax += 1
    if 'cv_fr' in metrics:  # coefficient of variation of firing rates hist
        cv_fr_ax = fig.add_subplot(nrows, ncols, n_cur_ax)
        cv_fr = cv_fr_hist(units_b, units=clusters, hist_win=fr_hist_win, fr_win=fr_ma_win,
                           n_cv_bins=n_cv_bins, bins=bins, ax=cv_fr_ax)
        m['cv_fr'] = cv_fr
        n_cur_ax += 1
    if 'spks_missed' in metrics:  # fraction missing spikes hist
        spks_missed_ax = fig.add_subplot(nrows, ncols, n_cur_ax)
        fraction_missing = spks_missed_hist(
            units_b, units=clusters, bins=bins, spks_per_bin=spks_per_bin, sigma=sigma,
            ax=spks_missed_ax)
        m['fraction_missing'] = fraction_missing
        n_cur_ax += 1
    if 'isi_viol' in metrics:  # fraction isi violations hist
        isi_viol_ax = fig.add_subplot(nrows, ncols, n_cur_ax)
        isi_viol = isi_viol_hist(units_b, units=clusters, rp=rp, bins=bins, ax=isi_viol_ax)
        m['isi_viol'] = isi_viol
        n_cur_ax += 1
    if 'max_drift' in metrics:  # max_drift hist
        max_drift_ax = fig.add_subplot(nrows, ncols, n_cur_ax)
        max_drift = max_drift_hist(units_b, units=clusters, bins=bins, ax=max_drift_ax)
        m['max_drift'] = max_drift
        n_cur_ax += 1
    if 'cum_drift' in metrics:  # cum_drift hist
        cum_drift_ax = fig.add_subplot(nrows, ncols, n_cur_ax)
        cum_drift = cum_drift_hist(units_b, units=clusters, bins=bins, ax=cum_drift_ax)
        m['cum_drift'] = cum_drift
        n_cur_ax += 1
    
    fig.subplots_adjust(left=0.075, right=0.925, top=0.925, bottom=0.075, wspace=0.45, hspace=0.4)
    return fig, m

def um_selected_plots(clusters, metrics, units_b, alf_probe_path, ephys_file_path, m,
                      metrics_params, save_dir=None):
    '''
    Computes/creates metrics and plots in a figure for specified units in a recording session.

    Parameters
    ----------
    clusters : list
        The clusters for which to generate the metrics summary plots.
    metrics : list
        The selected metrics plots to generate for the `unit_metrics_selected` figure. Possible
        values can include: 
        's' : 
        'cv_fr' :
        'spks_missed' : 
        'isi_viol' :
        'amp_heatmap' :
    units_b : bunch
        A units bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all units.
    alf_probe_path : string
        The absolute path to an 'alf/probe' directory.
    ephys_file_path : string
        The path to the binary ephys file.
    m : bunch
        A bunch containing metrics as fields.
    metrics_params : dict
        Parameters used for the summary metrics figure:
            'bins' : int OR sequence OR string. 
                The number of bins (or the bins, or the method used to compute the bins) used for
                computing the histograms. (see `numpy.histogram_bin_edges`).
            'rp' : float 
                The refractory period (in s) of the unit
            'spks_per_bin' : int 
                The number of spikes per bin from which to compute the spike feature histogram for
                `spks_missed`.
            'sigma' : float
                The standard deviation for the gaussian kernel used to compute the pdf from the
                spike feature histogram for `spks_missed`.
            'n_ch' : int 
                The number of channels used in plotting the amplitude heatmap.
            'fr_hist_win' : float
                The time window (in s) to use for computing spike counts for the instantaneous
                firing rate.
            'fr_ma_win' : float
                The time window (in s) to use as a moving slider to compute the instantaneous
                firing rate.
            'n_cv_bins' : int
                The number of equally spaced bins in time in which to compute the coefficient of
                variation of the firing rate.
            'n_ch_probe' : int
                The total number of channels from the recording (The number of rows in the binary
                ephys file).
    save_dir : string
        The path to which to save generated figures. (if `None`, figures will not be automatically
        saved)

    Returns
    -------
    m : bunch
        A bunch containing metrics as fields.
    fig : figure
        A handle to the figure generated.

    See Also
    --------
    brainbox.metrics.metrics
    brainbox.plot.plot
    
    Examples
    --------
    '''

    # Extract parameter values #
    #--------------------------#
    bins = metrics_params['bins']
    rp = metrics_params['rp']
    spks_per_bin = metrics_params['spks_per_bin']
    sigma = metrics_params['sigma']
    n_ch = metrics_params['n_ch']
    fr_hist_win = metrics_params['fr_hist_win']
    fr_ma_win = metrics_params['fr_ma_win']
    n_cv_bins = metrics_params['n_cv_bins']
    n_ch_probe = metrics_params['n_ch_probe']
    isi_win = metrics_params['isi_win']

    # Set figure #
    #------------#
    nrows = len(metrics)  # units will be in columns, and different features in rows
    ncols = len(clusters)
    fig = plt.figure(figsize=[16,8])
    fig.set_tight_layout(False)
    fig.suptitle('Selected Units Metrics')
    n_cur_ax = 1
    
    # Get alf objects for this session (needed for some metrics calculations below)
    clstrs_b = aio.load_object(alf_probe_path, 'clusters')    

    if 'isi_viol' in metrics:  # isi histogram
        for unit in clusters:
            cur_ax = fig.add_subplot(nrows, ncols, n_cur_ax)
            ts = units_b['times'][str(unit)]
            _, _, isis = bb.metrics.isi_viol(ts, rp=rp)
            isis = isis[np.where(isis < 0.1)[0]]
            cur_ax.set_xlim([0, isi_win])
            cur_ax.hist(isis, bins=bins)
            cur_ax.set_xlabel('ISI Time (s)')
            cur_ax.set_ylabel('Count')
            cur_ax.set_title('ISI Histogram for Unit {}'.format(unit))
            n_cur_ax += 1 
    if 'spks_missed' in metrics:  # pdf of missing spikes plot
        for unit in clusters:
            cur_ax = fig.add_subplot(nrows, ncols, n_cur_ax)
            amps = units_b['amps'][str(unit)]
            try:
                bb.plot.feat_cutoff(amps, 'amps', spks_per_bin=spks_per_bin, sigma=sigma,
                                    ax=[cur_ax])
            except AssertionError as e:
                print(e)
                print('Skipping this unit. (unit {})'.format(unit))
            n_cur_ax += 1
    if 'cv_fr' in metrics:  # coefficient of variation of firing rates plot
        for unit in clusters:
            cur_ax = fig.add_subplot(nrows, ncols, n_cur_ax)
            ts = units_b['times'][str(unit)]
            bb.plot.firing_rate(ts, hist_win=fr_hist_win, fr_win=fr_ma_win, n_bins=n_cv_bins,
                                ax=cur_ax)
            n_cur_ax += 1
    if 'amp_heatmap' in metrics:  # amplitude heatmap
        for unit in clusters:
            cur_ax = fig.add_subplot(nrows, ncols, n_cur_ax)
            ts = units_b['times'][str(unit)]
            max_ch = clstrs_b['channels'][1]
            if max_ch < n_ch:  # take only channels greater than `max_ch`.
                ch = np.arange(max_ch, max_ch + n_ch)
            elif (max_ch + n_ch) > n_ch_probe:  # take only channels less than `max_ch`.
                ch = np.arange(max_ch - n_ch, max_ch)
            else:  # take `n_c_ch` around `max_ch`.
                ch = np.arange(max_ch - (n_ch // 2), max_ch + (n_ch // 2))
            bb.plot.amp_heatmap(ephys_file_path, ts, ch, ax=cur_ax)
            n_cur_ax += 1
    # TODO add this
    if 'peth' in metrics:  # peth
        pass
    # TODO put `s` in a separate figure.
    if 's' in metrics:  # waveforms plot
        pass

    fig.subplots_adjust(left=0.075, right=0.925, top=0.925, bottom=0.075, wspace=0.4, hspace=0.5)
    return fig, m

def s_hist(ephys_file, units_b, clstrs_b, units=None, n_spks=100, n_ch=10, sr=30000,
           n_ch_probe=385, dtype='int16', car=False, bins='auto', ax=None):
    '''
    Plots a histogram of 's' (the spatiotemporal similarity of two sets of waveforms, for the first
    and last `n_spks` waveforms of a unit) for all `units`. 

    Parameters
    ----------
    ephys_file : string
        The file path to the binary ephys data.
    units_b : bunch
        A units bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all units.
    clstrs_b : bunch
        A clusters bunch containing fields with cluster information (e.g. amp, ch of max amp, depth
        of ch of max amp, etc.) for all clusters.
    units : ndarray
        The units for which to calculate 's' and plot in the historgram. (if `None`, histogram
        is created for all clusters)
    n_ch : int (optional)
        The number of channels around the channel of max amplitude to use to calculate 's'.
    n_spks : int (optional)
        The max first and last number of spikes to take to calculate 's'.
    sr : int (optional)
        The sampling rate (in hz) that the ephys data was acquired at.
    n_ch_probe : int (optional)
        The number of channels of the recording.
    dtype : str (optional)
        The datatype represented by the bytes in `ephys_file`.
    car : bool (optional)
        A flag to perform common-average-referencing before extracting waveforms.
    bins : int OR sequence OR string
        The number of bins used in computing the histograms. Can be a string, which specifies
        the method to use to compute the optimal number of bins (see `numpy.histogram_bin_edges`).
    ax : axessubplot (optional) 
        The axis handle to plot the histogram on. (if `None`, a new figure and axis is created)

    Returns
    -------
    s : ndarray
        The s values for each unit.

    See Also
    --------
    metrics.wf_similarity

    Examples
    --------
    '''
    
    # Get units.
    if units is None:  # we're using all units
        units = list(units_b['times'].keys())

    # Calculate 's'.
    s = np.ones(len(units),)
    for unit in range(len(units)):
        # Get the channel of max amplitude and `n_ch` around it.
        # If empty unit returned by spike sorter, create a NaN placeholder and skip it:
        if not(str(type(units_b['times'][str(unit)])) == "<class 'numpy.ndarray'>"):
            s[unit] = np.nan
            continue
        ts1 = units_b['times'][str(unit)][:n_spks]
        ts2 = units_b['times'][str(unit)][-n_spks:]
        max_ch = clstrs_b['channels'][unit]
        n_c_ch = n_ch // 2
        if max_ch < n_c_ch:  # take only channels greater than `max_ch`.
            ch = np.arange(max_ch, max_ch + n_ch)
        elif (max_ch + n_c_ch) > n_ch_probe:  # take only channels less than `max_ch`.
            ch = np.arange(max_ch - n_ch, max_ch)
        else:  # take `n_c_ch` around `max_ch`.
            ch = np.arange(max_ch - n_c_ch, max_ch + n_c_ch)
        # Extract the waveforms for these timestamps and compute similarity score.
        wf1 = bb.io.extract_waveforms(ephys_file, ts1, ch, sr=sr, n_ch_probe=n_ch_probe,
                                      dtype=dtype, car=car)
        wf2 = bb.io.extract_waveforms(ephys_file, ts2, ch, sr=sr, n_ch_probe=n_ch_probe,
                                      dtype=dtype, car=car)
        s[unit] = bb.metrics.wf_similarity(wf1, wf2)

    # Plot histogram.
    if ax is None:
        ax = plt.gca()

    ax.hist(s, bins)
    ax.set_title("'S' Values Histogram")
    ax.set_xlabel("'S'")
    ax.set_ylabel('Count')
    
    return s


def cv_fr_hist(units_b, units=None, hist_win=0.01, fr_win=0.05, n_cv_bins=10, bins='auto',
               ax=None):
    '''
    Plots a histogram of coefficient of variation of firing rate for all `units`.

    Parameters
    ----------
    units_b : bunch
        A units bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all units.
    units : ndarray
        The units for which to calculate 's' and plot in the historgram. (if `None`, histogram
        is created for all clusters)
    hist_win : float
        The time window (in s) to use for computing spike counts for the instantaneous
        firing rate.
    fr_win : float
        The time window (in s) to use as a moving slider to compute the instantaneous
        firing rate.
    n_cv_bins : int
        The number of equally spaced bins in time in which to compute the coefficient of
        variation of the firing rate.
    bins : int OR sequence OR string
        The number of bins used in computing the histograms. Can be a string, which specifies
        the method to use to compute the optimal number of bins (see `numpy.histogram_bin_edges`).
    ax : axessubplot (optional)
        The axis handle to plot the histogram on. (if `None`, a new figure and axis is created)

    Returns
    -------
    cv_fr : ndarray
        The coefficient of variation of firing rate values for each unit.

    See Also
    --------
    metrics.firing_rate_coeff_var

    Examples
    --------
    '''
    
    # Get units.
    if units is None:  # we're using all units
        units = list(units_b['times'].keys())
    
    # Calculate coefficient of variation of firing rate.
    cv_fr = np.ones(len(units),)
    for unit in range(len(units)):
        # If empty unit returned by spike sorter, create a NaN placeholder and skip it:
        if not(str(type(units_b['times'][str(unit)])) == "<class 'numpy.ndarray'>"):
            cv_fr[unit] = np.nan
            continue
        ts = units_b['times'][str(unit)]
        cv_fr[unit], _, _ = bb.metrics.firing_rate_coeff_var(ts, hist_win=hist_win, fr_win=fr_win,
                                                             n_bins=n_cv_bins)
    
    # Plot histogram.
    if ax is None:
        ax = plt.gca()

    ax.hist(cv_fr, bins)
    ax.set_title("Coefficient of Variation of Firing Rate Histogram")
    ax.set_xlabel("Coefficient of Variation of Firing Rate")
    ax.set_ylabel('Count')
    
    return cv_fr


def spks_missed_hist(units_b, units=None, spks_per_bin=20, sigma=5, bins='auto', ax=None):
    '''
    Plots a histogram of the approximate fraction of spikes missing from a spike feature
    distribution (assuming the distribution is symmetric) for all `units`.

    Parameters
    ----------
    units_b : bunch
        A units bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all units.
    units : ndarray
        The units for which to calculate 's' and plot in the historgram. (if `None`, histogram
        is created for all clusters)
    spks_per_bin : int (optional)
        The number of spikes per bin from which to compute the spike feature histogram.
    sigma : int (optional)
        The standard deviation for the gaussian kernel used to compute the pdf from the spike
        feature histogram.
    bins : int OR sequence OR string
        The number of bins used in computing the histograms. Can be a string, which specifies
        the method to use to compute the optimal number of bins (see `numpy.histogram_bin_edges`).
    ax : axessubplot (optional)
        The axis handle to plot the histogram on. (if `None`, a new figure and axis is created)

    Returns
    -------
    frac_missing : ndarray
        The fraction of missing spikes.

    See Also
    --------
    metrics.feat_cutoff

    Examples
    --------
    '''
    
    # Get units.
    if units is None:  # we're using all units
        units = list(units_b['times'].keys())
    
    # Calculate fraction of missing spikes for each unit.
    frac_missing = np.ones(len(units),)
    for unit in range(len(units)):
        # If empty unit returned by spike sorter, create a NaN placeholder and skip it:
        if not(str(type(units_b['times'][str(unit)])) == "<class 'numpy.ndarray'>"):
            frac_missing[unit] = np.nan
            continue
        try:  # need a minimum number of spikes for `feat_cutoff`
            amps = units_b['amps'][str(unit)]
            frac_missing[unit], _, _ = bb.metrics.feat_cutoff(
                amps, spks_per_bin=spks_per_bin, sigma=sigma)
        except:  # if didn't meet min num spikes requirement, set as nan
            frac_missing[unit] = np.nan    
    
    # Plot histogram.
    if ax is None:
        ax = plt.gca()

    ax.hist(frac_missing, bins)
    ax.set_title("Fraction of Missing Spikes Histogram")
    ax.set_xlabel("Fraction of Missing Spikes")
    ax.set_ylabel('Count')
    
    return frac_missing


def isi_viol_hist(units_b, units=None, rp=0.002, bins='auto', ax=None):
    '''
    Plots a histogram of fraction of isi violations for all `units`.

    Parameters
    ----------
    units_b : bunch
        A units bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all units.
    units : ndarray
        The units for which to calculate 's' and plot in the historgram. (if `None`, histogram
        is created for all clusters)
    rp : float
        The refractory period (in s).
    bins : int OR sequence OR string
        The number of bins used in computing the histograms. Can be a string, which specifies
        the method to use to compute the optimal number of bins (see `numpy.histogram_bin_edges`).
    ax : axessubplot (optional)
        The axis handle to plot the histogram on. (if `None`, a new figure and axis is created)

    Returns
    -------
    frac_isi_viol : ndarray
        The fraction of isi violations for each unit.

    See Also
    --------
    metrics.isi_viol

    Examples
    --------
    '''
    
    # Get units.
    if units is None:  # we're using all units
        units = list(units_b['times'].keys())
    
    # Calculate fraction of isi violations for each unit.
    frac_isi_viol = np.ones(len(units),)
    for unit in range(len(units)):
        # If empty unit returned by spike sorter, create a NaN placeholder and skip it:
        if not(str(type(units_b['times'][str(unit)])) == "<class 'numpy.ndarray'>"):
            frac_isi_viol[unit] = np.nan
            continue
        ts = units_b['times'][str(unit)]
        frac_isi_viol[unit], _, _ = bb.metrics.isi_viol(ts, rp=rp)
    
    # Plot histogram.
    if ax is None:
        ax = plt.gca()

    ax.hist(frac_isi_viol, bins)
    ax.set_title("Fraction of ISI Violations Histogram")
    ax.set_xlabel("Fraction of ISI Violations")
    ax.set_ylabel('Count')
    
    return frac_isi_viol


def max_drift_hist(units_b, units=None, bins='auto', ax=None):
    '''
    Plots a histogram of the maximum drift values for all `units`.

    Parameters
    ----------
    units_b : bunch
        A units bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all units.
    units : ndarray
        The units for which to calculate 's' and plot in the historgram. (if `None`, histogram
        is created for all clusters)
    bins : int OR sequence OR string
        The number of bins used in computing the histograms. Can be a string, which specifies
        the method to use to compute the optimal number of bins (see `numpy.histogram_bin_edges`).
    ax : axessubplot (optional)
        The axis handle to plot the histogram on. (if `None`, a new figure and axis is created)

    Returns
    -------
    md : ndarray
        The max drift values for each unit.

    See Also
    --------
    metrics.max_drift

    Examples
    --------
    '''
    
    # Get units.
    if units is None:  # we're using all units
        units = list(units_b['times'].keys())
    
    # Calculate fraction of isi violations for each unit.
    md = np.ones(len(units),)
    for unit in range(len(units)):
        # If empty unit returned by spike sorter, create a NaN placeholder and skip it:
        if not(str(type(units_b['times'][str(unit)])) == "<class 'numpy.ndarray'>"):
            md[unit] = np.nan
            continue
        depths = units_b['depths'][str(unit)]
        md[unit] = bb.metrics.max_drift(depths)
    
    # Plot histogram.
    if ax is None:
        ax = plt.gca()

    ax.hist(md, bins)
    ax.set_title("Max Drift Values Histogram")
    ax.set_xlabel("Max Drift (mm)")
    ax.set_ylabel('Count')
    
    return md


def cum_drift_hist(units_b, units=None, bins='auto', ax=None):
    '''
    Plots a histogram of the cumulative drift values for all `units`.

    Parameters
    ----------
    units_b : bunch
        A units bunch containing fields with spike information (e.g. cluster IDs, times, features,
        etc.) for all units.
    units : ndarray
        The units for which to calculate 's' and plot in the historgram. (if `None`, histogram
        is created for all clusters)
    bins : int OR sequence OR string
        The number of bins used in computing the histograms. Can be a string, which specifies
        the method to use to compute the optimal number of bins (see `numpy.histogram_bin_edges`).
    ax : axessubplot (optional)
        The axis handle to plot the histogram on. (if `None`, a new figure and axis is created)

    Returns
    -------
    cd : ndarray
        The cumulative drift values for each unit.

    See Also
    --------
    metrics.cum_drift

    Examples
    --------
    '''


    # Get units.
    if units is None:  # we're using all units
        units = list(units_b['times'].keys())
    
    # Calculate fraction of isi violations for each unit.
    cd = np.ones(len(units),)
    for unit in range(len(units)):
        # If empty unit returned by spike sorter, create a NaN placeholder and skip it:
        if not(str(type(units_b['times'][str(unit)])) == "<class 'numpy.ndarray'>"):
            cd[unit] = np.nan
            continue
        depths = units_b['depths'][str(unit)]
        cd[unit] = bb.metrics.cum_drift(depths)
    
    # Plot histogram.
    if ax is None:
        ax = plt.gca()

    ax.hist(cd, bins)
    ax.set_title("Cumulative Drift Values Histogram")
    ax.set_xlabel("Cumulative Drift (mm)")
    ax.set_ylabel('Count')
    
    return cd
