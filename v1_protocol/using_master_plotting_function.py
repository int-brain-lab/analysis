'''
Complete instructions for using the master plotting function, `plot.gen_figures`.

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
    conda install conda==4.7.12  # first update conda to 4.7.12
    cd ~\int-brain-lab\ibllib
    conda env create --name v1_cert --file brainbox_env.yml
    conda activate v1_cert
    pip install -r requirements.txt
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

    In python (ensure your python path can access the code):

    ```
    cd ~\int-brain-lab
    import os
    import sys
    sys.path.extend([os.path.join(os.getcwd(), 'ibllib'), os.path.join(os.getcwd(), 'analysis')])
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
        'clusters.probes',
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
        '_iblrig_taskData.raw',
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
'''

# Ensure the python path is set correctly
from pathlib import Path
import os
import sys
sys.path.extend([os.path.join(Path.home(), 'int-brain-lab', 'ibllib'),
                 os.path.join(Path.home(), 'int-brain-lab', 'analysis')])
from oneibl.one import ONE
from v1_protocol import plot as v1_plot

# Set the eid as `eid` and probe name as `probe` - these two input args are required for running
# `gen_figures`
one = ONE()
eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
probe = 'probe_right'

# If not already downloaded, download all data for a particular recording session.
one.load(eid, dataset_types=one.list(), clobber=False, download_only=True)

# Recreate figures for 'ZM_2104/19-09-19/001' session.
m, cluster_sets, _ = v1_plot.gen_figures(
    eid, probe, n_selected_cl=4,
    grating_response_selected=True, unit_metrics_selected=True, 
    auto_filt_cl_params={'min_amp': 0, 'min_fr': 0, 'max_fpr': 100, 'rp': 0.002})