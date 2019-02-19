# Anne Urai, CSHL, 2018
# see https://github.com/int-brain-lab/ibllib/tree/master/python/oneibl/examples

import time, os
from datetime import timedelta
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed as shell

# IBL stuff
from oneibl.one import ONE

# loading and plotting functions
from define_paths import fig_path
# from behavior_plots import *
# from load_mouse_data import * # this has all plotting functions

## INITIALIZE A FEW THINGS
sns.set_style("darkgrid", {'xtick.bottom': True,'ytick.left': True, 'lines.markeredgewidth':0 } )
sns.set_context(context="paper")
sns.set_palette("colorblind") # palette for water types

## CONNECT TO ONE
one = ONE() # initialize

# get folder to save plots
path = fig_path()
path = os.path.join(path, 'per_lab/')
if not os.path.exists(path):
    os.mkdir(path)

users = ['churchlandlab', 'mainenlab', 'wittenlab', 'cortexlab', 'angelakilab']

# ============================================= #
# START BIG OVERVIEW PLOT
# ============================================= #

for lidx, lab in enumerate(users):

    print(lab)

    # LOAD ALL AMBIENT SENSOR DATA FOR THIS LAB
    eids = one.search(dataset_types='_iblrig_ambientSensorData.raw', lab=lab)
    for eid in eids:
        asd = one.load(eid, dataset_types=['_iblrig_ambientSensorData.raw'])
        if not 'ambient' in locals():
            ambient = pd.DataFrame.from_dict(asd)
        else:
            ambient = ambient.append(pd.DataFrame.from_dict(asd), sort=False, ignore_index=True)

    print(ambient.describe())

    # fig  = plt.figure(figsize=(11.69, 8.27), constrained_layout=True)
    # axes = []

    # # SAVE FIGURE PER BATCH
    # fig.suptitle('Ambient sensor, %s' %(lab))
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig.savefig(os.path.join(path + '%s_ambientsensor.pdf'%lab))
    # plt.close(fig)

