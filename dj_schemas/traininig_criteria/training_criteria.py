"""
Plot full psychometric functions as a function of choice history,
and separately for 20/80 and 80/20 blocks
"""

import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import seaborn as sns
import datajoint as dj
from IPython import embed as shell # for debugging
from scipy.special import erf # for psychometric functions
import datetime

## INITIALIZE A FEW THINGS
sns.set(style="darkgrid", context="paper", font='Arial')
sns.set(style="darkgrid", context="paper")
sns.set(style="darkgrid", context="paper", font_scale=1.3)

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses
from dj_tools import *

figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
# GRAB ALL DATA FROM DATAJOINT
# ================================= #

b = behavioral_analyses.PsychResults * behavior.TrialSet * \
	(subject.Subject & 'subject_birth_date > "2018-09-01"') * subject.SubjectLab

behav = pd.DataFrame(b.proj('subject_nickname', 'n_trials',
	'lab_name', 'session_start_time', 'performance_easy', 
	'threshold', 'bias', 'lapse_low', 'lapse_high').fetch(as_dict=True))

"""
Change inclusion criteria as follows:
On unbiased blocks:
Bias: from 16% contrast to 10% contrast
Percentage correct in each individual session: from 80% to 90% correct
Lapse rate on either side averaged across last three sessions: from 20% to 10%
Cap on median RT: less than 2 sec at the lowest contrast (no capt right now)

On biased blocks:
Lapse rate on either side averaged across last three sessions: 10% (Currently not being checked)
"""

def find_trained_3days(df, easy_crit):

	outp = pd.DataFrame({'istrained':False, 'trained_date':np.nan, 'sessions_to_trained':np.nan}, index=[0])
	perf = df['performance_easy']

	for i in np.arange(2, len(perf)):
		if np.all(perf[i-2:i] > easy_crit) & np.all(df.n_trials.iloc[i-2:i] > 400):
			outp.istrained       = True
			outp.trained_date    = df.session_start_time.iloc[i]
			outp.sessions_to_trained = i

	return outp

for lapse in [0.8, 0.85, 0.9]:
	trained_dates = behav.groupby(['subject_nickname', 'lab_name']).apply(find_trained_3days, 
		easy_crit=lapse).reset_index()
	print(trained_dates.istrained.mean())
	# print(trained_dates.sessions_to_trained.mean())

	fig = sns.swarmplot(x="lab_name", y="sessions_to_trained", data=trained_dates)
	fig.set_title('%f percent of all mice trained'%(trained_dates.istrained.mean()*100))
	plt.savefig(os.path.join(figpath, "days_to_trained_easy%d.pdf"%(lapse*100)))
	plt.close('all')
