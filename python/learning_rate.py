#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:52:40 2019

@author: guido
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:30:25 2018

@author: guido
"""

import os
import pandas as pd
import numpy as np
from oneibl.one import ONE
from define_paths import analysis_path
from load_mouse_data import get_water_weight, get_behavior
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy as sp
from os.path import join

def pf(x, alpha, beta, gamma):
    return 0.5 + (1 - 0.5 - gamma) * (1. / (1 + np.exp( -(x-alpha)/beta)))

## CONNECT TO ONE
one = ONE() # initialize
ac = one._alyxClient

# get a list of all mice 
#subjects = pd.DataFrame(one.alyx.get('/subjects?responsible_user=jeanpaul'))
#subjects = pd.DataFrame(one.alyx.get('/subjects?nickname=DY_001'))
#subjects = pd.DataFrame(one.alyx.get('/subjects?&alive=True&stock=False'))
#subjects = pd.DataFrame(one.alyx.get('/subjects?alive=True?&responsible_user=ines'))
subjects = pd.DataFrame(one.alyx.get('/subjects'))

# get folder to save plots
path = analysis_path()
if not os.path.exists(path):
    os.mkdir(path)
    
print(subjects['nickname'].unique())
    
learning = pd.DataFrame(columns=['mouse','lab','delay','rise','asymp','date_start'])

for i, mouse in enumerate(subjects['nickname']):
    print('Loading data for ' + mouse)
    try:
        # load in data
        behav = get_behavior(mouse)
        weight_water, baseline = get_water_weight(mouse)  
        sub = ac.rest('subjects','read',mouse)
    except:
        continue
    
    if len(np.unique(behav['days'])) < 5:
        continue
    
    # calculate perc correct on easy trials and reaction times
    behav['correct_easy'] = behav.correct
    behav.loc[np.abs(behav['signedContrast']) < 50, 'correct_easy'] = np.NaN
    perf = behav.groupby(['date'])['correct_easy'].mean().reset_index()
    days = behav.groupby(['date'])['days'].median().reset_index()
    perf['days'] = days['days']
    trial_num = behav.groupby(['date'])['correct_easy'].size().reset_index()
    perf['trial_num'] = trial_num['correct_easy']
    
    # Remove weird first session
    if perf['days'][1]-perf['days'][0] > 20:
        perf = perf[1:]
        perf['days'] = perf['days'] - perf['days'][1]
    
    delay = np.nan
    rise = np.nan
    asymp = np.nan
    if sum(perf['correct_easy'] > 0.75) > 2:
        # Fit function
        par0 = sp.array([20, 5, 0])
        lms = sp.array([[0,0,0],[100,10,0.5]])
        try:
            par, mcov = curve_fit(pf, perf['days'], perf['correct_easy'], par0, bounds=lms)
        except:
            continue
        vec_x = np.arange(0,max(perf['days'])+2,0.1)
        fit_x = pf(vec_x, par[0], par[1], par[2])
        norm_x = (fit_x-min(fit_x))/max(fit_x-min(fit_x))
        delay = vec_x[np.argmin(np.abs(norm_x - 0.2))]
        rise = vec_x[np.argmin(np.abs(norm_x - 0.8))]
        asymp = vec_x[np.argmin(np.abs(norm_x - 0.99))]
           
        plt.figure()
        plt.plot(perf['days'], perf['correct_easy'], 'ro')
        plt.plot(np.arange(0,max(perf['days'])+2,0.1), pf(np.arange(0,max(perf['days'])+2,0.1), par[0], par[1], par[2]), linewidth=2)
        plt.title(mouse)
        plt.xlabel('Training sessions (days)')
        plt.ylabel('Performance (% correct easy trials)')
        plt.plot([-2,delay], [pf(delay, par[0], par[1], par[2]), pf(delay, par[0], par[1], par[2])], '--' ,color=[0.5,0.5,0.5])
        plt.plot([delay,delay], [pf(delay, par[0], par[1], par[2]), 0.4], '--' ,color=[0.5,0.5,0.5])
        plt.plot([-2,rise], [pf(rise, par[0], par[1], par[2]), pf(rise, par[0], par[1], par[2])], '--' ,color=[0.5,0.5,0.5])
        plt.plot([rise,rise], [pf(rise, par[0], par[1], par[2]), 0.4], '--' ,color=[0.5,0.5,0.5])
        plt.plot([-2,asymp], [pf(asymp, par[0], par[1], par[2]), pf(asymp, par[0], par[1], par[2])], '--' ,color=[0.5,0.5,0.5])
        plt.plot([asymp,asymp], [pf(asymp, par[0], par[1], par[2]), 0.4], '--' ,color=[0.5,0.5,0.5])
        plt.savefig(join(path,'Learning','%s.pdf'%mouse))
        #plt.show()
        plt.close()
        
    learning.loc[i] = [mouse, subjects.loc[i,'lab'], delay, rise, asymp, perf.loc[perf.index.values[0],'date']]
learning.to_pickle(join(path,'learning_rates'))


    
    
    
    