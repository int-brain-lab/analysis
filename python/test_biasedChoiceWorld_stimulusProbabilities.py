# Anne Urai, CSHL, 2018
# see https://github.com/int-brain-lab/ibllib/tree/master/python/oneibl/examples

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
subjects     = pd.DataFrame(one.alyx.get('/subjects?&nickname=IBL_36'))

# get folder to save plots
path = fig_path()
path = os.path.join(path, 'per_mouse/')
if not os.path.exists(path):
    os.mkdir(path)

print(subjects['nickname'].unique())

for i, mouse in enumerate(subjects['nickname']):

    # SKIP IF THIS FIGURE ALREADY EXISTS, DONT OVERWRITE
    if os.path.exists(os.path.join(path + '%s_overview_test.pdf'%mouse)):
         pass #continue
    print(mouse)

    behav = get_behavior(mouse, date_range=['2019-02-11', '2019-02-13'])
    shell()


   behav['stimulusSide'] = np.sign(behav['signedContrast'])
   choicedat = behav.groupby('probabilityLeft').agg({'trial':'count', 
    'stimulusSide':'mean', 'signedContrast':'mean', 'contrastLeft':'mean', 'contrastRight':'mean'}).reset_index()