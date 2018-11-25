# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:39:52 2018

@author: Miles
"""

import psychofit as psy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#from matplotlib.dates import MONDAY
import psychofit as psy # https://github.com/cortex-lab/psychofit
import seaborn as sns 
import pandas as pd
from IPython import embed as shell

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def fit_psychfunc(df):
    choicedat = df.groupby('signedContrast').agg({'trial':'max', 'choice2':'mean'}).reset_index()
    pars, L = psy.mle_fit_psycho(choicedat.values.transpose(), P_model='erf_psycho_2gammas', 
        parstart=np.array([choicedat['signedContrast'].mean(), 20., 0.05, 0.05]), 
        parmin=np.array([choicedat['signedContrast'].min(), 0., 0., 0.]), 
        parmax=np.array([choicedat['signedContrast'].max(), 100., 1, 1]))
    df2 = {'bias':pars[0],'threshold':pars[1], 'lapselow':pars[2], 'lapsehigh':pars[3]}

    return pd.DataFrame(df2, index=[0])

def plot_psychometric(df, ax=None, color="black"):
    """
    Plots psychometric data for a given DataFrame of behavioural trials
    
    If the data contains more than six different contrasts (or > three per side)
    the data are fit with an erf function.  The x-axis is percent contrast and 
    the y-axis is the proportion of 'rightward choices', i.e. trials where the 
    subject turned the wheel clockwise to threshold.
    
    Example:
        df = alf.load_behaviour('2018-09-11_1_Mouse1', r'\\server\SubjectData')
        plot_psychometric(df)
        
    Args:
        df (DataFrame): DataFrame constructed from an ALF trials object.
        ax (Axes): Axes to plot to.  If None, a new figure is created.
        
    Returns:
        ax (Axes): The plot axes
    """
    
    if len(df['signedContrast'].unique()) > 4:
        df2 = df.groupby(['signedContrast']).agg({'choice':'count', 'choice2':'mean'}).reset_index()
        df2.rename(columns={"choice2": "fraction", "choice": "ntrials"}, inplace=True)

        pars, L = psy.mle_fit_psycho(df2.transpose().values, # extract the data from the df
                                     P_model='erf_psycho_2gammas',
                                     parstart=np.array([df2['signedContrast'].mean(), 20., 0.05, 0.05]),
                                     parmin=np.array([df2['signedContrast'].min(), 0., 0., 0.]), 
                                     parmax=np.array([df2['signedContrast'].max(), 100., 1, 1]))
        sns.lineplot(np.arange(-100,100), psy.erf_psycho_2gammas( pars, np.arange(-100,100)), color=color, ax=ax)

    # plot datapoints on top
    sns.lineplot(x='signedContrast', y='choice2', err_style="bars", linewidth=0, linestyle='None', mew=0.5,
        marker='.', ci=68, data=df, color=color, ax=ax)

    # Reduce the clutter
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(['-100', '-50', '0', '50', '100'])
    ax.set_yticks([0, .5, 1])
    # Set the limits
    ax.set_xlim([-110, 110])
    ax.set_ylim([-0.03, 1.03])
    ax.set_xlabel('Contrast (%)')

    return ax


def plot_chronometric(df, ax, color):

    sns.lineplot(x='signedContrast', y='rt', err_style="bars", mew=0.5,
        estimator=np.median, marker='.', ci=68, data=df, color=color, ax=ax)
    ax.set(xlabel="Contrast (%)", ylabel="RT (s)")
    ax.grid(True)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(['-100', '-50', '0', '50', '100'])


def fix_date_axis(ax):
    # deal with date axis and make nice looking 
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    for item in ax.get_xticklabels():
        item.set_rotation(60)

