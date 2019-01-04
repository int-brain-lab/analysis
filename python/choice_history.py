# Anne Urai, CSHL, 2019

import time, os, datetime
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed as shell

# IBL stuff
from oneibl.one import ONE

# loading and plotting functions
from define_paths import fig_path
from os.path import join as join
from behavior_plots import *
from load_mouse_data import get_water_weight, get_behavior

# ============================================= #
# START BIG OVERVIEW PLOT
# ============================================= #

## INITIALIZE A FEW THINGS
sns.set_style("darkgrid", {'xtick.bottom': True,'ytick.left': True, 'lines.markeredgewidth':0 } )
sns.set_context(context="paper")

## CONNECT TO ONE
one = ONE() # initialize

# get a list of all mice that are currently training
subjects     = pd.DataFrame(one.alyx.get('/subjects?&responsible_user=valeria'))
print(subjects['nickname'].unique())

# ============================================= #
# GET DATA

for i, mouse in enumerate(subjects['nickname']):
    print(mouse)
    if i == 0:
        behav = get_behavior(mouse)
    else:
        behav.append(get_behavior(mouse))

# SELECT THOSE SESSIONS WHERE ALL CONTRASTS WERE PRESENTED
numcontrasts = 5

# ============================================= #
# PLOT PSYCHOMETRIC FUNCTIONS

fig, axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True)
