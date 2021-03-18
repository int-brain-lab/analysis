'''
Complete instructions and examples for using the master plotting function, `plot.gen_figures`.

Below are instructions for ensuring Python has access to the appropriate code for running:

    1) Clone the directories and checkout the 'brainbox' branch of the 'ibllib' repository.
    *Note, if you have already cloned the 'analysis' and 'ibllib' repositories, then create a new
    directory in your home directory, 'int-brain-lab', and move the 'analysis' and 'ibllib'
    directories here. Else, in your git/OS terminal, run:

    ```
    cd ~  # navigate to your home directory
    mkdir int-brain-lab  # make a new directory, 'int-brain-lab'
    cd int-brain-lab  # navigate inside 'int-brain-lab'
    git clone https://github.com/int-brain-lab/ibllib
    git clone https://github.com/int-brain-lab/analysis
    cd ibllib
    git checkout --track origin/brainbox  # OR, if it already exists, run `git checkout brainbox`
    ```

    2) Create a conda environment in which you'll run the code. In your conda/OS terminal, run:

    ```
    conda update conda  # first update to latest version of conda
    cd ~\int-brain-lab\ibllib
    conda env create --name v1_cert --file brainbox_env.yml  # create env from `brainbox_env` file
    conda activate v1_cert  # activate env
    pip install -r requirements.txt  # install necessary non-conda packages
    conda update jupyter_console  # update jupyter_console to be compatible with latest ipython
    ```

Below are the git and conda commands you should run before each time you run the master plotting
function to ensure you are using the correct code:

    ```
    # activate the correct env
    conda activate v1_cert
    # checkout and pull ibllib@brainbox and analysis@master
    cd ~\int-brain-lab\ibllib
    git checkout brainbox
    git fetch
    git pull
    cd cd ~\int-brain-lab\analysis
    git checkout master
    git fetch
    git pull
    ```

Below are the dataset_types required depending on the plots/metrics to be generated.

    *Note, although the default call to the master plotting function will not require access to the
    raw data, it is still recommended to download *all* data for a particular eid when possible:
        ```
        from oneibl.one import ONE
        one = ONE()
        # get eid
        eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
        # download ALL data
        one.load(eid, dataset_types=one.list(), clobber=False, download_only=True)
        ```

    Default (required for the default function call and all other function calls):
        'clusters.amps',
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'clusters.peakToTrough',
        'clusters.uuids',
        'clusters.waveforms',
        'clusters.waveformsChannels',
        'spikes.amps',
        'spikes.clusters',
        'spikes.depths',
        'spikes.samples',
        'spikes.templates',
        'spikes.times'

    Required for stimulus info extraction and to generate grating response figures:
        'ephysData.raw.meta',
        '_spikeglx_sync.channels',
        '_spikeglx_sync.polarities',
        '_spikeglx_sync.times',
        '_iblrig_RFMapStim.raw',
        '_iblrig_taskSettings.raw',
        '_iblrig_codeFiles.raw'

    Required for metrics/plots that need access to the raw data:
        'ephysData.raw.ap',
        'ephysData.raw.ch',
        'ephysData.raw.lf',
        'ephysData.raw.meta',
        'ephysData.raw.nidq',
        'ephysData.raw.sync',
        'ephysData.raw.timestamps',
        'ephysData.raw.wiring'

Current sessions in examples: 'ZM_2104/2019-09-1/1', 'KS003/2019-11-1/1',
'CSK-scan-008/2019-12-0/8', 'CSHL_020/2019-12-03/001'

*Note: see the master plotting function (`analysis\v1_protocol\plot.gen_figures`) for detailed
documentation on parametrizing all the possible metrics/plots.

*Note: When saving figures, the `fig_names` input argument should be a dict with keys for the
figures, and values as the name of the image file (NOT the full path).The keys in `fig_names`
MUST be amongst the following:
    'um_summary' : The name for the summary metrics figure.
    'um_selected' : The name for the selected units' metrics figure.
    'gr_summary' : The name for the summary grating response summary figure.
    'gr_selected' : The name for the selected units' grating response figure.
See 'Example 2' and 'Example 3' for examples of saving figures.
'''

# Ensure the python path is set correctly
from pathlib import Path
import os
import sys
sys.path.extend([os.path.join(Path.home(), 'int-brain-lab', 'ibllib'),
                 os.path.join(Path.home(), 'int-brain-lab', 'analysis')])
import numpy as np
from oneibl.one import ONE
from v1_protocol import plot as v1_plot
import alf.io as aio
import brainbox as bb

# The examples below can be run independently.

# Example 1: For 'ZM_2104/2019-09-19/001' 'probe_right', generate all 4 figures (grating response
# summary, grating response selected, unit metrics summary, and unit metrics selected) using
# default parameters. For the summary figures, use all units, and for the selected figures, use
# 4 randomly chosen units.
# -------------------------------------------------------------------------------------------------

# Set the eid as `eid` and probe name as `probe` - these two input args are required for running
# `gen_figures`
one = ONE()
eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
probe = 'probe_right'

# Get paths to the required dataset_types. If required dataset_types are not already downloaded,
# download them.
dtypes = [
        'clusters.amps',
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'clusters.peakToTrough',
        'clusters.uuids',
        'clusters.waveforms',
        'clusters.waveformsChannels',
        'spikes.amps',
        'spikes.clusters',
        'spikes.depths',
        'spikes.samples',
        'spikes.templates',
        'spikes.times',
        'ephysData.raw.meta',
        '_spikeglx_sync.channels',
        '_spikeglx_sync.polarities',
        '_spikeglx_sync.times',
        '_iblrig_RFMapStim.raw',
        '_iblrig_taskSettings.raw',
        '_iblrig_codeFiles.raw'
        ]
d_paths = one.load(eid, dataset_types=dtypes, clobber=False, download_only=True)

# Call master plotting function.
# The below choice of `auto_filt_cl_params` ensures that all units are included.
m, cluster_sets, _ = v1_plot.gen_figures(
    eid, probe, n_selected_cl=4,
    grating_response_selected=True, unit_metrics_selected=True,
    filt_params={'min_amp': 0, 'min_fr': 0, 'max_fpr': 100, 'rp': 0.002})


# Example 2: For 'KS003/2019-11-25/001' 'probe01', generate just the unit metrics summary and unit
# metrics selected figures. Generate the summary figure for all units, and generate the selected
# figures (in batches of 5) for all units with a minimum amplitude > 40 uV and a minimum firing
# rate > 1 Hz. Save all figures in the home 'v1cert_figs' directory.
# -------------------------------------------------------------------------------------------------

# Set the eid as `eid` and probe name as `probe` - these two input args are required for running
# `gen_figures`
one = ONE()
eid = one.search(subject='KS003', date='2019-11-25', number=1)[0]
probe = 'probe01'

# Get paths to the required dataset_types. If required dataset_types are not already downloaded,
# download them.
dtypes = [
        'clusters.amps',
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'clusters.peakToTrough',
        'clusters.uuids',
        'clusters.waveforms',
        'clusters.waveformsChannels',
        'spikes.amps',
        'spikes.clusters',
        'spikes.depths',
        'spikes.samples',
        'spikes.templates',
        'spikes.times',
        ]
d_paths = one.load(eid, dataset_types=dtypes, clobber=False, download_only=True)

# Filter units
alf_probe_path = os.path.split(d_paths[0])[0]
spks_b = aio.load_object(alf_probe_path, 'spikes')
units_b = bb.processing.get_units_bunch(spks_b)
T = spks_b.times[-1] - spks_b.times[0]  # length of recording session
filt_units = bb.processing.filter_units(units_b, T, min_amp=50e-6, min_fr=3)

# Specify parameters for saving figures.
save_dir = Path.joinpath(Path.home(), 'v1cert_figs')
# Set filename of figures to save
fig_names = {'um_summary': 'KS003_2019-11-19_1_summary'}

# Call master plotting function for metrics summary figure.
m, cluster_sets, fig_list = v1_plot.gen_figures(
    eid, probe, cluster_ids_selected=filt_units, extract_stim_info=False,
    unit_metrics_summary=True, unit_metrics_selected=False, grating_response_summary=False,
    grating_response_selected=False, save_dir=save_dir, fig_names=fig_names)

# Call master plotting function in a loop on filtered units to generated selected figures for units
# in batches of 5.

batch_sz = 5  # number of units per figure
n_i = int(np.ceil(len(filt_units) / batch_sz))  # number of iterations in for loop
cur_unit = 0
for i in range(n_i):
    fig_names = {'um_selected': 'KS003_2019-11-19_1_selected_' + str(i)}
    m, cluster_sets, fig_list = v1_plot.gen_figures(
        eid, probe, cluster_ids_selected=filt_units[cur_unit:(cur_unit + batch_sz)],
        extract_stim_info=False, unit_metrics_summary=False, unit_metrics_selected=True,
        grating_response_summary=False, grating_response_selected=False,
        save_dir=save_dir, fig_names=fig_names)
    cur_unit += batch_sz


# Example 3: For 'CSK-scan-008/2019-12-09/008' generate just the grating response figures. For
# grating response parameters: change the time shown before and after the grating onset to each be
# 1 s, change the bin width for determining spikes/bin to 10 ms, and the bin width of the smoothing
# kernel to 50 ms. For receptive field parameters: change the bin width to 10 ms, the number of
# bins for calculating the receptive fields to 10, and the number of depths to aggregate clusters
# to 20. Save all figures in the home 'v1cert_figs' directory.
#
# -------------------------------------------------------------------------------------------------

# Set the eid as `eid` and probe name as `probe` - these two input args are required for running
# `gen_figures`
one = ONE()
eid = one.search(subject='CSK-scan-008', date='2019-12-09', number=8)[0]
probe = 'probe00'

# Get paths to the required dataset_types. If required dataset_types are not already downloaded,
# download them.
dtypes = [
        'clusters.amps',
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'clusters.peakToTrough',
        'clusters.uuids',
        'clusters.waveforms',
        'clusters.waveformsChannels',
        'spikes.amps',
        'spikes.clusters',
        'spikes.depths',
        'spikes.samples',
        'spikes.templates',
        'spikes.times',
        'ephysData.raw.meta',
        '_spikeglx_sync.channels',
        '_spikeglx_sync.polarities',
        '_spikeglx_sync.times',
        '_iblrig_RFMapStim.raw',
        '_iblrig_taskSettings.raw',
        '_iblrig_codeFiles.raw'
        ]
d_paths = one.load(eid, dataset_types=dtypes, clobber=False, download_only=True)

# Specify parameters for saving figures.
save_dir = Path.joinpath(Path.home(), 'v1cert_figs')
# Set filename of figures to save
fig_names = {'gr_summary': 'CSK-scan-008_2019-12-09_008_grsummary',
             'gr_selected': 'CSK-scan-008_2019-12-09_008_grselected'}

# Call master plotting with appropriate input args to get parametrized grating response figures:
m, cluster_sets, _ = v1_plot.gen_figures(
    eid, probe, grating_response_summary=True, grating_response_selected=True,
    unit_metrics_summary=False, unit_metrics_selected=False,
    grating_response_params={'pre_t': 1, 'post_t': 1, 'bin_t': 0.01, 'sigma': 0.05},
    rf_params={'method': 'corr', 'binsize': 0.01, 'lags': 10, 'n_depths': 20, 'use_svd': False},
    save_dir=save_dir, fig_names=fig_names)


# Example 4: For 'CSHL_020/2019-12-03/001' 'probe00' generate the grating response summary and unit
# metrics summary figures only. Filter the units used in these figures to only include those with
# a minimum amplitude of 60 uV and no minimum firing rate. Add the coefficient of firing rate plot
# to the default plots, with parameters that specify that the time window for computing firing rate
# spike counts is 500 ms, and the moving average to compute instantaneous firing rate is 2 s.
# -------------------------------------------------------------------------------------------------

# Set the eid as `eid` and probe name as `probe` - these two input args are required for running
# `gen_figures`
one = ONE()
eid = one.search(subject='CSHL_020', date='2019-12-03', number=1)[0]
probe = 'probe00'

# Get paths to the required dataset_types. If required dataset_types are not already downloaded,
# download them.
dtypes = [
        'clusters.amps',
        'clusters.channels',
        'clusters.depths',
        'clusters.metrics',
        'clusters.peakToTrough',
        'clusters.uuids',
        'clusters.waveforms',
        'clusters.waveformsChannels',
        'spikes.amps',
        'spikes.clusters',
        'spikes.depths',
        'spikes.samples',
        'spikes.templates',
        'spikes.times',
        'ephysData.raw.meta',
        '_spikeglx_sync.channels',
        '_spikeglx_sync.polarities',
        '_spikeglx_sync.times',
        '_iblrig_RFMapStim.raw',
        '_iblrig_taskSettings.raw',
        '_iblrig_codeFiles.raw'
        ]
d_paths = one.load(eid, dataset_types=dtypes, clobber=False, download_only=True)

# Call master plotting with appropriate input args to get parametrized summary figures:
m, cluster_sets, _ = v1_plot.gen_figures(
    eid, probe, grating_response_summary=True, grating_response_selected=False,
    unit_metrics_summary=True, unit_metrics_selected=False,
    filt_params={'min_amp': 60e-6, 'min_fr': 0},
    summary_metrics = ['feat_vars', 'spks_missed', 'isi_viol', 'max_drift_depth',
                       'cum_drift_depth', 'max_drift_amp', 'cum_drift_amp', 'pres_ratio', 'cv_fr'],
    summary_metrics_params={'fr_hist_win': 0.5, 'fr_ma_win': 2})
