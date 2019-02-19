# Anne Urai, CSHL, 2018
# see https://github.com/int-brain-lab/ibllib/tree/master/python/oneibl/examples

import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# IBL stuff
from oneibl.one import ONE

# loading and plotting functions
from define_paths import fig_path
from behavior_plots import *
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

users = ['angelakilab', 'wittenlab', 'churchlandlab', 'mainenlab']

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

        if ambient_tmp.empty:
            continue
        
        # HACK: wait for Nicco to return values, rather than wrapping in dict
        ambient_tmp['Temperature_C']    = ambient_tmp['Temperature_C'].apply(pd.Series)
        ambient_tmp['RelativeHumidity'] = ambient_tmp['RelativeHumidity'].apply(pd.Series)
        ambient_tmp['AirPressure_mb']   = ambient_tmp['AirPressure_mb'].apply(pd.Series)

        # take values at the beginning and end of session
        ambient_summ = ambient_tmp.iloc[[2, -2]].reset_index()

        # add timing information per trial
        ambient_summ.loc[0, 'datetime'] = pd.to_datetime(details[ix]['start_time'])
        ambient_summ.loc[1, 'datetime'] = pd.to_datetime(details[ix]['end_time'])
        
        # add rig information per trial
        ambient_summ.loc[:, 'rig'] = details[ix]['location']
        
        # REMOVE TIMES THAT DON'T MAKE SENSE
        if any(ambient_summ['datetime'] > pd.to_datetime('today')) or any(ambient_summ['datetime'] < pd.to_datetime('2018-11-01')):
            continue

        # append to large dataframe
        if ix == 0:
            ambient = ambient_summ.copy()
        else:
            ambient = ambient.append(ambient_summ, sort=False, ignore_index=True)
        print('%d/%d'%(ix, len(eids)))
        
    # ambient.loc[ambient['datetime'] > pd.to_datetime('today'), 'datetime'] = np.NaN
    print(ambient.describe())

    # SAVE FIGURE SHOWING THESE RELATIONSHIPS
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(11.69, 8.27))
    
    # show each rig in its own color
    for color, group in ambient.groupby(['rig']):
        ax0.plot_date(group['datetime'], group['Temperature_C'], label=color)
        ax1.plot_date(group['datetime'], group['RelativeHumidity'], label=color)
        ax2.plot_date(group['datetime'], group['AirPressure_mb'], label=color)
        
    ax0.set_ylabel('Temperature (C)')
    ax1.set_ylabel('Relative humidity (%)')
    ax2.set_ylabel('Air pressure (mb)')

    # INDICATE RECOMMENDATIONS FROM JAX
    # Temperature and humidity:  Temperatures of 65-75°F (~18-23°C) with 40-60% humidity are recommended.
    ax2.legend()

    # deal with date axis and make nice looking 
    ax2.xaxis_date()
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    # ax2.xaxis.set_minor_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    for item in ax2.get_xticklabels():
        item.set_rotation(60)
        
    fig.suptitle('Ambient sensor, %s'%lab)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(path + '%s_ambientsensor.pdf'%lab))
    plt.close(fig)
    del ambient

