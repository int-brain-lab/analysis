# Anne Urai, CSHL, 2018
# see https://github.com/int-brain-lab/ibllib/tree/master/python/oneibl/examples

import time, os
from datetime import timedelta
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

users = ['churchlandlab', 'mainenlab', 'angelakilab', 'wittenlab', 'cortexlab']

# ============================================= #
# START BIG OVERVIEW PLOT
# ============================================= #

for lidx, lab in enumerate(users):

    print(lab)

    # LOAD ALL AMBIENT SENSOR DATA FOR THIS LAB
    # see https://github.com/int-brain-lab/ibllib/issues/51#event-2148508648
    eids, details = one.search(dataset_types='_iblrig_ambientSensorData.raw', lab=lab, details=True)

    for ix, eid in enumerate(eids):

        asd = one.load(eid, dataset_types=['_iblrig_ambientSensorData.raw'])
        ambient_tmp = pd.DataFrame(asd[0])

        # HACK: wait for Nicco to return values, rather than wrapping in dict
        ambient_tmp['Temperature_C']    = ambient_tmp['Temperature_C'].apply(pd.Series)
        ambient_tmp['RelativeHumidity'] = ambient_tmp['RelativeHumidity'].apply(pd.Series)
        ambient_tmp['AirPressure_mb']   = ambient_tmp['AirPressure_mb'].apply(pd.Series)

        # take values at the beginning and end of session
        ambient_summ = ambient_tmp.iloc[[0, -1]].reset_index()

        # add timing information per trial
        ambient_summ.loc[0, 'datetime'] = pd.to_datetime(details[ix]['start_time'])
        ambient_summ.loc[1, 'datetime'] = pd.to_datetime(details[ix]['end_time'])

        # append to large dataframe
        if not 'ambient' in locals():
            ambient = ambient_summ.copy()
        else:
            ambient = ambient.append(ambient_summ, sort=False, ignore_index=True)
        print('%d/%d'%(ix, len(eids)))
        
    # REMOVE TIMES IN THE FUTURE
    ambient.loc[ambient['datetime'] > pd.to_datetime('today'), 'datetime'] = np.NaN
    print(ambient.describe())

    # SAVE FIGURE SHOWING THESE RELATIONSHIPS
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(11.69, 8.27))
    ax0.plot_date(ambient.datetime, ambient.Temperature_C)
    ax0.set_ylabel('Temperature (C)')
    ax1.plot_date(ambient.datetime, ambient.RelativeHumidity)
    ax1.set_ylabel('Humidity')
    ax2.plot_date(ambient.datetime, ambient.AirPressure_mb)
    ax2.set_ylabel('Air Pressure (mb)')
    ax2.xaxis.set_major_locator(mpl.dates.DayLocator())
    ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b %d'))
    plt.xticks(rotation=-20)
    
    # SAVE
    fig.suptitle('Ambient sensor, %s'%lab)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(path + '%s_ambientsensor.pdf'%lab))
    plt.close(fig)

