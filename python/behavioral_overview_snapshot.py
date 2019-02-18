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
subjects     = pd.DataFrame(one.alyx.get('/subjects?&alive=True&stock=False'))
# subjects     = pd.DataFrame(one.alyx.get('/subjects?&nickname=IBL_36'))

# get folder to save plots
path = fig_path()
path = os.path.join(path, 'per_mouse/')
if not os.path.exists(path):
    os.mkdir(path)

print(subjects['nickname'].unique())

for i, mouse in enumerate(subjects['nickname']):

    try:

        # SKIP IF THIS FIGURE ALREADY EXISTS, DONT OVERWRITE
        if os.path.exists(os.path.join(path + '%s_overview_test.pdf'%mouse)):
             pass #continue
        print(mouse)

        # MAKE THE FIGURE, divide subplots using gridspec
        fig, axes = plt.subplots(ncols=5, nrows=4, constrained_layout=True,
            gridspec_kw=dict(width_ratios=[2,2,1,1,1], height_ratios=[1,1,1,1]), figsize=(11.69, 8.27))
        sns.set_palette("colorblind") # palette for water types

        # ============================================= #
        # GET DATA
        # ============================================= #

        weight_water, baseline = get_water_weight(mouse)   

        fig.suptitle('Mouse %s (%s), born %s, user %s (%s) \nstrain %s, cage %s, %s' %(subjects['nickname'][i],
         subjects['sex'][i], subjects['birth_date'][i],
         subjects['responsible_user'][i], subjects['lab'][i],
         subjects['strain'][i], subjects['litter'][i], subjects['description'][i]))
        
        # ============================================ #
        # WEIGHT CURVE AND WATER INTAKE
        # ============================================= #
        
        xlims = [weight_water.date.min()-datetime.timedelta(days=2), weight_water.date.max()+datetime.timedelta(days=2)]
        plot_water_weight_curve(weight_water, baseline, axes[0,0])

        # ============================================= #
        # TRIAL COUNTS AND SESSION DURATION
        # ============================================= #
        
        behav = get_behavior(mouse)
        xlims = [behav.date.min()-datetime.timedelta(days=1), behav.date.max()+datetime.timedelta(days=1)]
        plot_trialcounts_sessionlength(behav, axes[1,0], xlims)

        # ============================================= #
        # PERFORMANCE AND MEDIAN RT
        # ============================================= #

        plot_performance_rt(behav, axes[2,0], xlims)

        # ============================================= #
        # CONTRAST/CHOICE HEATMAP
        # ============================================= #

        plot_contrast_heatmap(behav, axes[3,0])

        # ============================================= #
        # PSYCHOMETRIC FUNCTION FITS OVER TIME
        # ============================================= #

        # fit psychfunc on choice fraction, rather than identity
        pars = behav.groupby(['date', 'probabilityLeft_block']).apply(fit_psychfunc).reset_index()
        parsdict = {'threshold': r'Threshold $(\sigma)$', 'bias': r'Bias $(\mu)$',
            'lapselow': r'Lapse low $(\gamma)$', 'lapsehigh': r'Lapse high $(\lambda)$'}
        ylims = [[-5, 105], [-105, 105], [-0.05, 1.05], [-0.05, 1.05]]
        yticks = [[0, 19, 100], [-100, -16, 0, 16, 100], [-0, 0.2, 0.5, 1], [-0, 0.2, 0.5, 1]]

        # pick a good-looking diverging colormap (green/blueish to red/orange) with black in the middle
        cmap = sns.diverging_palette(20, 220, n=len(behav['probabilityLeft_block'].unique()), center="dark")
        if len(behav['probabilityLeft_block'].unique()) == 1:
            cmap = "gist_gray"
        sns.set_palette(cmap)

        # plot the fitted parameters
        for pidx, (var, labelname) in enumerate(parsdict.items()):
            ax = axes[pidx,1]
            sns.lineplot(x="date", y=var, marker='o', hue="probabilityLeft_block", linestyle='', lw=0,
                palette=cmap, data=pars, legend=None, ax=ax)
            ax.set(xlabel='', ylabel=labelname, ylim=ylims[pidx],
                yticks=yticks[pidx],
                xlim=[behav.date.min()-datetime.timedelta(days=1), behav.date.max()+datetime.timedelta(days=1)])

            fix_date_axis(ax)
            if pidx == 0:
                ax.set(title=r'$\gamma + (1 -\gamma-\lambda)  (erf(\frac{x-\mu}{\sigma} + 1)/2$')

        # ============================================= #
        # LAST THREE SESSIONS
        # ============================================= #

        didx = 1
        sorteddays = behav['days'].sort_values(ascending=True).unique()
        for day in behav['days'].unique():

            # use only the last three days
            if (len(sorteddays) < 3 or day < sorteddays[-3]):
                continue

            # grab only that day
            dat = behav.loc[behav['days'] == day, :]
            print(dat['date'].unique())
            didx += 1

            # colormap for the asymmetric blocks
            cmap = sns.diverging_palette(20, 220, n=len(dat['probabilityLeft_block'].unique()), center="dark")
            if len(dat['probabilityLeft_block'].unique()) == 1:
                cmap = [np.array([0,0,0,1])]

            # PSYCHOMETRIC FUNCTION
            ax = axes[0, didx]
            for ix, probLeft in enumerate(dat['probabilityLeft_block'].sort_values().unique()):
                plot_psychometric(dat.loc[dat['probabilityLeft_block'] == probLeft, :], ax=ax, color=cmap[ix])
            ax.set(xlabel="Contrast (%)", ylabel="Choose right (%)")
            ax.set(title=pd.to_datetime(dat['start_time'].unique()[0]).strftime('%b-%d, %A'))

            # CHRONOMETRIC FUNCTION
            ax = axes[1, didx]
            for ix, probLeft in enumerate(dat['probabilityLeft_block'].sort_values().unique()):
                plot_chronometric(dat.loc[dat['probabilityLeft_block'] == probLeft, :], ax, cmap[ix])
            ax.set(ylim=[0.1,1.5], yticks=[0.1, 1.5])
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: 
                ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

            # RTS THROUGHOUT SESSION
            ax = axes[2, didx]
            sns.scatterplot(x='trial', y='rt', style='correct', hue='correct',
                palette={1:"#009E73", 0:"#D55E00"}, # from https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mpl-data/stylelib/seaborn-colorblind.mplstyle
                markers={1:'o', 0:'X'}, s=10, edgecolors='face',
                alpha=.5, data=dat, ax=ax, legend=False)
            # running median overlaid
            sns.lineplot(x='trial', y='rt', color='black', ci=None,
                data=dat[['trial', 'rt']].rolling(10).median(), ax=ax)
            ax.set(xlabel="Trial number", ylabel="RT (s)", ylim=[0.02, 60])
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos:
                ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

        # clean up layout
        for i in range(3):
            axes[i,3].set(ylabel='')
            axes[i,4].set(ylabel='')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(path + '%s_overview_test.pdf'%mouse))
        plt.close(fig)

    except:
        print("%s failed to run" %mouse)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(path + '%s_overview_test.pdf'%mouse))
        plt.close(fig)
        pass
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # fig.savefig(os.path.join(path + '%s_overview_test.pdf'%mouse))
        # plt.close(fig)



