'''

Complete instructions for using the master plotting function, `plot.gen_figures`.

Below are 2 options for ensuring Python has access to the appropriate code for running
`plot.gen_figures`.

Option A) Manually add the directories to your Python path in Python.

    1) Clone the directories and checkout the 'brainbox' branch of the 'ibllib' repository.
    *Note, if you have already cloned the 'analysis' and 'ibllib' repositories, then create a new 
    directory in your home directory, 'int-brain-lab', and move the 'analysis' and 'ibllib'
    directories here. Else, in your git/OS terminal, run:

    ```
    cd ~  # navigate to your home directory
    mkdir int-brain-lab  # make a new directory, 'int-brain-lab'
    cd int-brain-lab  # navigate inside 'int-brain-lab'
    git clone https://github.com/int-brain-lab/ibllib
    git clone https://github.com/int-brain-lab/
    cd ibllib
    git checkout --track origin/brainbox  # OR, if it already exists, run `git checkout brainbox`
    ```

    2) Create a conda environment in which you'll run the code. In your conda/OS terminal, run:

    ```
    cd ~/int-brain-lab/ibllib
    conda env create --name v1_cert --file brainbox_env.yml
    conda activate v1_cert
    ```

    3) In Python, add these folders to your path. In Python, run:

    ```
    cd ~/int-brain-lab
    import os
    import sys
    sys.path.extend([os.path.join(os.getcwd(), 'ibllib'), os.path.join(os.getcwd(), 'analysis')])
    ```
    
Option B) pip install 'ibllib'.

    1) - 2) Follow instructions 1-2 above.

    3) Install 'ibllib' as an editable package. In your conda/OS terminal, run:

    ```
    cd ~/int-brain-lab/ibllib
    conda activate v1_cert  # ensure you are in the 'v1_cert' environment
    pip install -r requirements.txt
    pip install -e . git+https://github.com/int-brain-lab/ibllib.git@brainbox
    pip install -U git+https://github.com/cortex-lab/phylib.git@master
    ```
'''

# %% Recreate figures for 'ZM_2104/19-09-19/001' session.
from oneibl.one import ONE
from v1_protocol import plot as v1_plot
one = ONE()
eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
probe = 'probe_right'
m, cluster_sets, _ = v1_plot.gen_figures(
    eid, 'probe_right', n_selected_cl=4,
    grating_response_selected=True, unit_metrics_selected=True, 
    auto_filt_cl_params={'min_amp': 0, 'min_fr': 0, 'max_fpr': 100, 'rp': 0.002})
