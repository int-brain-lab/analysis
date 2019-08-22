

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed as shell # for debugging
from scipy.optimize import curve_fit

# ================================================================== #
# DEFINE LEARNING CURVES
# FROM GUIDO https://github.com/int-brain-lab/analysis/blob/master/python/learning_rate.py
# PAPER: BATHELLIER ET AL https://www.pnas.org/content/110/49/19950
# ================================================================== #

def sigmoid(x, start, alpha, beta, asymp):
    return start + (1 - start - asymp) * (1. / (1 + np.exp( -(x-alpha)/beta)))

def fit_learningcurve(df):

    # only fit learning curves if there is data from a sufficient number of days
    if len(df) >= 21:

        # fit the actual function
        par, pcov = curve_fit(sigmoid, df['session_day'], df['performance_easy'],
                              sp.array([20, 5, 0]), bounds=sp.array([[0, 0, 0], [100, 10, 0.5]]))

        # compute parameter estimates around these values
        perr = np.sqrt(np.diag(pcov))

        # get some more parameters out
        vec_x = np.arange(0, max(df['session_day']) + 2, 0.1)
        fit_x = sigmoid(vec_x, par[0], par[1], par[2])
        perf   = max(fit_x) # asymptotic performance for this animal

        if perf > 0.7:
            norm_x = (fit_x - min(fit_x)) / (max(fit_x) - min(fit_x)) # normalize learning to the asymptote
            rise_time  = vec_x[np.argmin(np.abs(norm_x - 0.2))] # how long to reach 20% of performance?
            delay_time  = vec_x[np.argmin(np.abs(norm_x - 0.8))] # after delay, how long to reach 80% of performance?
            asymp_time  = vec_x[np.argmin(np.abs(norm_x - 0.99))] # how long to reach asymptotic performance?
        else: # if there is no appreciable learning curve, these learning times don't make much sense
            delay = np.nan
            rise = np.nan
            asymp = np.nan
        
        df2 = pd.DataFrame({'delay_time': delay_time, 'rise_time': rise_time, 'asymp_time':_time, 
            'max_performance': perf, 'start':par[0], 
            'alpha': par[1], 'beta': par[2], 'asymp': par[3]}, 
            'start_perr':perr[0], 'alpha_perr':perr[1], 'beta_perr':perr[2], 'asymp_perr':perr[3],
            index=[0])

    else:
        df2 = pd.DataFrame({'delay_time': np.nan, 'rise_time': np.nan, 'asymp_time': np.nan,
                            'max_performance':np.nan, 'start':np.nan, 'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan,
                            'start_perr':np.nan, 'alpha_perr':np.nan, 'beta_perr':np.nan, 'asymp_perr':np.nan,
                            }, index=[0])
        print('cannot fit learning curves with fewer than 21 days of training')

    return df2


def plot_learningcurve(x, y, subj, **kwargs):

    # summary stats - average psychfunc over observers
    df = pd.DataFrame({'session_day':x, 'performance_easy':y, 'subject_nickname':subj})

    # fit learning curve
    pars  = fit_learningcurve(df)
    vec_x = np.arange(0, max(df.session_day) + 2, 0.1)
    fit_x = sigmoid(vec_x, float(pars.alpha), float(pars.beta), float(pars.gamma))
    
    # plot lines at 20 and 80 % points
    # USE SEABORN, AX.PLOT WILL BREAK FACETGRID!
    if len(subj.unique()) == 1:
        sns.lineplot([0, pars.delay_time.item()], [sigmoid(pars.delay_time.item(), float(pars.start), float(pars.alpha), float(pars.beta), float(pars.asymp)),
                    sigmoid(pars.delay_time.item(), float(pars.start), float(pars.alpha), float(pars.beta), float(pars.asymp))],
                    style=0, dashes={0:(2,1)}, lw=1, legend=False, **kwargs)
        sns.lineplot([pars.delay_time.item(), pars.delay_time.item()], [0.4, sigmoid(pars.delay_time.item(), float(pars.start), float(pars.alpha), float(pars.beta), float(pars.asymp))],
                    style=0, dashes={0:(2,1)}, lw=1, legend=False, **kwargs)
        sns.lineplot([0, pars.rise_time.item()], [sigmoid(pars.rise_time.item(), loat(pars.start), float(pars.alpha), float(pars.beta), float(pars.asymp)),
                    sigmoid(pars.rise_time.item(), loat(pars.start), float(pars.alpha), float(pars.beta), float(pars.asymp))],
                     style=0, dashes={0:(2,1)}, legend=False, **kwargs)
        sns.lineplot([pars.rise_time.item(), rise_time.rise_time.item()], [0.4, sigmoid(pars.rise_time.item(), loat(pars.start), float(pars.alpha), float(pars.beta), float(pars.asymp))],
                    style=0, dashes={0:(2,1)}, legend=False, **kwargs)

    # plot fitted function
    sns.lineplot(vec_x, fit_x, linewidth=2, **kwargs)
    
    # plot datapoints with errorbars (across subjects) on top
    sns.lineplot(df['session_day'], df['performance_easy'], err_style="bars", linewidth=0, 
                 linestyle='None', mew=0.5, marker='o', ci=68, markersize=6, **kwargs)


    