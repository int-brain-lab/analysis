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
from psychofit import psychofit as psy # https://github.com/cortex-lab/psychofit
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
    
    TODO Process three response types
    TODO Better titling of figure
    TODO Return fit pars if available
    TODO May as well reuse perf_per_contrast?
    TODO: Change plot_psychometric to split by side prob
    """
    
    contrastSet = np.sort(df['signedContrast'].unique())
    #choiceSet = np.array(set(df['choice']))
    nn = np.array([sum((df['signedContrast']==c) & (df['included']==True)) for c in contrastSet])
    pp = np.array([sum((df['signedContrast']==c) & (df['included']==True) & (df['choice']==1)) for c in contrastSet])/nn
    # ci = 1.96*np.sqrt(pp*(1-pp)/nn) # TODO: this is not the binomial CI
    
    # define binomial CI
    def binom_interval(success, total, confint=0.95):
        quantile = (1 - confint) / 2
        lower = sp.stats.beta.ppf(quantile, success, total - success + 1)
        upper = sp.stats.beta.ppf(1 - quantile, success + 1, total - success)
        return (lower, upper)

    lowerci, upperci = binom_interval(pp*nn, nn)

    if contrastSet.size > 4:
        pars, L = psy.mle_fit_psycho(np.vstack((contrastSet,nn,pp)), 
                                     P_model='erf_psycho_2gammas',
                                     parstart=np.array([np.mean(contrastSet), 3., 0.05, 0.05]),
                                     parmin=np.array([np.min(contrastSet), 10., 0., 0.]), 
                                     parmax=np.array([np.max(contrastSet), 30., .4, .4]))
        sns.lineplot(np.arange(-100,100), psy.erf_psycho_2gammas( pars, np.arange(-100,100)), color=color, ax=ax)

    # when there are not enough contrasts, still fit the same errorbar
    ax.errorbar(contrastSet, pp, pp-lowerci, upperci-pp, fmt='o', ecolor=color, mfc=color, mec="white")

    # Reduce the clutter
    ax.set_xticks([-100, -50, -25, -12.5, -6, 0, 6, 12.5, 25, 50, 100])
    ax.set_xticklabels(['-100', '', '', '', '', '0', '', '', '', '', '100'])
    ax.set_yticks([0, .5, 1])
    # Set the limits
    ax.set_xlim([-110, 110])
    ax.set_ylim([-0.03, 1.03])
    ax.set_xlabel('Contrast (%)')

    return ax


def plot_chronometric(df, ax, color):

    contrastSet = np.sort(df['signedContrast'].unique())
    df2 = df.groupby(['signedContrast']).agg({'rt':'median'}).reset_index()

    # get quantiles of the RT distribution
    def q1(x):
        return x.quantile(0.25)

    def q2(x):
        return x.quantile(0.75)
    f = {'rt': [q1,q2]}
    qlow = df.groupby(['signedContrast']).agg(f).reset_index()

    # sns.pointplot(x="signedContrast", y="rt", color=color, estimator=np.median, ci=None, join=True, data=df, ax=ax)
    ax.errorbar(df2['signedContrast'], df2['rt'], df2['rt']-qlow['rt']['q1'], 
        qlow['rt']['q2']-df2['rt'], 'o-', color=color, mec="white")
    ax.set(xlabel="Contrast (%)", ylabel="RT (s)")
    ax.grid(True)
    ax.set_xticks([-100, -50, -25, -12.5, -6, 0, 6, 12.5, 25, 50, 100])
    ax.set_xticklabels(['-100', '', '', '', '', '0', '', '', '', '', '100'])

def plot_perf_heatmap(dfs, ax=None):
    """
    Plots a heat-map of performance for each contrast per session.
    
    The x-axis is the contrast, going from highest contrast on the left to 
    highest contrast on the right. The y-axis is the session number, ordered 
    from most recent.  
        
    Example:
        refs, date, seq = dat.list_exps('Mouse1', rootDir = r'\\server\Data')
        dfs = [load_behaviour(ref[0]) for ref in refs]
        plot_perf_heatmap(dfs)
        
    Args:
        dfs (List): List of data frames constructed from an ALF trials object.
        ax (Axes): Axes to plot to.  If None, a new figure is created.
        
    Returns:
        ax (Axes): The plot axes
    
    TODO: Optional contrast set input
    """

    if ax is None:
        plt.figure()
        ax = plt.gca()

    import copy; cmap=copy.copy(plt.get_cmap('vlag'))
    cmap.set_bad(color="w") # remove those squares

    if not isinstance(dfs, (list,)):

        # Anne's version
        # TODO: only take the mean when there is more than 1 trial (to remove bug in early sessions)
        pp  = dfs.groupby(['signedContrast', 'days']).agg({'choice2':'mean'}).reset_index()
        pp2 = pp.pivot("signedContrast", "days",  "choice2").sort_values(by='signedContrast', ascending=False)
        pp2 = pp2.reindex([-100, -50, -25, -12, -6, 0, 6, 12, 25, 50, 100])

        # inset axes for colorbar
        axins1 = inset_axes(ax, width="5%", height="9q0%", loc='right',
            bbox_to_anchor=(0.1, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0,)
        # now heatmap
        sns.heatmap(pp2, linewidths=.5, ax=ax, vmin=0, vmax=1, cmap=cmap, cbar=True,
            cbar_ax=axins1,
            cbar_kws={'label': 'Choose right (%)', 'shrink': 0.8, 'ticks': []})
        ax.set(ylabel="Contrast (%)")

        # fix the date axis
        dates  = dfs.date.unique()
        xpos   = np.arange(len(dates)) + 0.5 # the tick locations for each day
        xticks = [i for i, dt in enumerate(dates) if pd.to_datetime(dt).weekday() is 0]
        ax.set_xticks(np.array(xticks) + 0.5)

        xticklabels = [pd.to_datetime(dt).strftime('%b-%d') for i, dt in enumerate(dates) if pd.to_datetime(dt).weekday() is 0]
        ax.set_xticklabels(xticklabels)
        for item in ax.get_xticklabels():
            item.set_rotation(60)

    else:
        # Miles' version
        pp = np.vstack([perf_per_contrast(df) for df in dfs])
        pp = np.ma.array(pp, mask=np.isnan(pp))

        ax.imshow(pp, extent=[0, 1, 0, 1], cmap=cmap, vmin = 0, vmax = 1)
        ax.set_xticks([0.05, .5, 0.95])
        ax.set_xticklabels([-100, 0, 100])
        ax.set_yticks(list(range(0,pp.shape[0],-1)))
        ax.set_yticklabels(list(range(0,pp.shape[0],-1)))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # Set bounds of axes lines
    #ax.spines['left'].set_bounds(0, 1)
    #ax.spines['bottom'].set_bounds(0, len(df.index))
    # Explode out axes
    #ax.spines['left'].set_position(('outward',10))
    #ax.spines['bottom'].set_position(('outward',10))
    return ax


def fix_date_axis(ax):
    # deal with date axis and make nice looking 
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    for item in ax.get_xticklabels():
        item.set_rotation(60)

