

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


def sigmoid(x, alpha, beta, gamma):
    return 0.5 + (1 - 0.5 - gamma) * (1. / (1 + np.exp( -(x-alpha)/beta)))

def fit_learningcurve(df, subject_nickname):

    # only fit learning curves if there is data from a sufficient number of days
    if len(df) > 14:

        # fit the actual function
        par, mcov = curve_fit(sigmoid, df['session_day'], df['performance_easy'],
                              sp.array([20, 5, 0]), bounds=sp.array([[0, 0, 0], [100, 10, 0.5]]))

        # get some more parameters out
        vec_x = np.arange(0, max(df['session_day']) + 2, 0.1)
        fit_x = sigmoid(vec_x, par[0], par[1], par[2])

        perf  = max(fit_x) # asymptotic performance for this animal
        norm_x = (fit_x - min(fit_x)) / max(fit_x - min(fit_x)) # normalize learning to the asymptote
        delay  = vec_x[np.argmin(np.abs(norm_x - 0.2))] # how long to reach 20% of performance?
        rise   = vec_x[np.argmin(np.abs(norm_x - 0.8))] - delay # after delay, how long to reach 80% of performance?
        asymp   = vec_x[np.argmin(np.abs(norm_x - 0.99))] - delay # how long to reach asymptotic performance?
        
        df2 = pd.DataFrame({'delay': delay, 'rise': rise, 'asymp': asymp, 'perf': perf, 'alpha': par[0], 'beta': par[1], 'gamma': par[2]}, 
                            index=[subject_nickname])
    else:
        df2 = pd.DataFrame({'delay': np.nan, 'rise': np.nan, 'asymp': np.nan, 'perf':np.nan, 'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan}, 
                           index=[subject_nickname])

    return df2


def plot_learningcurve(x, y, subj, ax):

    # summary stats - average psychfunc over observers
    df = pd.DataFrame({'session_day':x, 'performance_easy':y, 'subject_nickname':subj})

    # fit learning curve
    pars = fit_learningcurve(df, subj)
    vec_x = np.arange(0, max(df.session_day) + 2, 0.1)
    fit_x = sigmoid(vec_x, float(pars.alpha), float(pars.beta), float(pars.gamma))
    
    # plot lines at 20 and 80 % points
    norm_x = (fit_x - min(fit_x)) / max(fit_x - min(fit_x))
    delay  = vec_x[np.argmin(np.abs(norm_x - 0.2))]
    rise   = vec_x[np.argmin(np.abs(norm_x - 0.8))]
    ax.plot([0,delay,delay], [sigmoid(delay, float(pars.alpha), float(pars.beta), float(pars.gamma)),
            sigmoid(delay, float(pars.alpha), float(pars.beta), float(pars.gamma)), 0.4], 'k--')
    ax.plot([0,rise,rise], [sigmoid(rise, float(pars.alpha), float(pars.beta), float(pars.gamma)),
            sigmoid(rise, float(pars.alpha), float(pars.beta), float(pars.gamma)), 0.4], 'k--')
    
    # plot fitted function
    sns.lineplot(vec_x, fit_x, color="#e74c3c", linewidth=2, ax=ax)
    
    # plot datapoints with errorbars on top
    sns.lineplot(df['session_day'], df['performance_easy'], err_style="bars", linewidth=0, 
                 linestyle='None', mew=0.5, marker='o', ci=68, markersize=6, color='#34495e', ax=ax)


    