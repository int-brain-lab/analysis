"""
Creates summary metrics and plots for units in a recording session.

This module assumes that the 'analysis', 'iblscripts', and 'ibllib' repositories from
https://github.com/int-brain-lab are directories in the current working folder. These directories
should be on the 'certification', 'certification', 'brainbox' branches, respectively.

This module assumes that the required data for a particular eid is already saved in the CACHE_DIR
specified by `.one_params` (the default location to which ONE saves data when running the `load`
method). 

To download *all* data for a particular eid:
    `from oneibl.one import ONE`
    `one = ONE()`
    # get eid
    `eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]`
    # download data
    one.load(eid, dataset_types=one.list(), clobber=False, download_only=True)

List of required data (alf objects) depending on the figures to be generated:
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

Run this as a script from within python (navigate to this directory and run):
`exec(open('plot.py').read())`
or in a terminal, outside of python (navigate to this directory and run):
`python plot.py`

"""

import sys
# Add `ibllib` and `iblscripts` to path.
sys.path.extend(['.\\ibllib', '.\\iblscripts'])
from pathlib import Path
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import pandas as pd
from oneibl.one import ONE
import alf.io
import brainbox as bb
from brainbox.processing import bincount2D
# Import code for extracting stimulus info to get v1 certification files (`_iblcertif_`).
from deploy.serverpc.certification import certification_pipeline


def gen_figures(eid, clusters=[], grating_response_summary=True, grating_response_selected=False,
                unit_metrics_summary=True, unit_metrics_selected=False, extract_stim_info=True):
    '''
    Generates figures for the V1 certification protocol.

    Parameters
    ----------
    eid : string
        The experiment ID for a given recording session: the UUID of the session as per Alyx.
    clusters : array-like
        The clusters for which to generate `grating_response_ind` and/or `unit_metrics_ind`.
    grating_response_summary : bool
        A flag for returning a figure with summary grating response plots based on all units.
    grating_response_selected : bool
        A flag for returning a figure with grating response plots for the selected units in 
        `clusters`.
    unit_metrics_summary : bool
        A flag for returning a figure with summary metrics plots based on all units.
    unit_metrics_selected : bool
        A flag for returning a figure with single unit metrics plots for the selected units in
        `clusters`.
    extract_stim_info : bool
        A flag for extracting stimulus info from the recording session into an alf directory.

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
    '''
    
    # Get necessary data via ONE
    one = ONE()
    dtypes = []
    if extract_stim_info:
        dtypes.extend([
            '_spikeglx_sync.channels',
            '_spikeglx_sync.polarities',
            '_spikeglx_sync.times',
            '_iblrig_RFMapStim.raw',
            '_iblrig_codeFiles.raw',
            '_iblrig_taskSettings.raw'])
        spikes_path = one.load(eid, dataset_types='spikes.amps', \
                               clobber=False, download_only=True)[0]
        alf_dir_part = np.where([part == 'alf' for part in Path(spikes_path).parts])[0][0]
        session_path = os.path.join(*Path(spikes_path).parts[:alf_dir_part])
        certification_pipeline.extract_stimulus_info_to_alf(session_path, save=True)


def summary_plots(eid):
    '''
    Computes metrics and creates plots for all units in a given recording session.

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

