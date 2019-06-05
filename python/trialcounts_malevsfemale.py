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
from matplotlib import dates

## INITIALIZE A FEW THINGS
sns.set_style("darkgrid")
sns.set_context(context="poster")

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from dj_tools import *
figpath  = os.path.join(os.path.expanduser('~'), 'Data/Figures_IBL')

# ================================= #
##  GET DATA
# ================================= #

b = (behavior.TrialSet) \
	* (acquisition.Session.proj(session_date='DATE(session_start_time)') & 'session_start_time > "2019-03-01"') \
	* (subject.Subject() & 'subject_birth_date between "2018-09-01" and "2019-02-01"') * subject.SubjectLab() \
	* action.Weighing.proj(weighing_date='DATE(weighing_time)')
bdat = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time'))

print(bdat['subject_nickname'].unique())
bdat['date_march'] = pd.to_timedelta(pd.to_datetime(bdat.session_date)).dt.total_seconds().astype(int)

g = sns.lmplot(x="date_march", y="n_trials", hue="sex", 
	col="lab_name", units="subject_nickname", 
	col_wrap=4, lowess=True, data=bdat)

# fig = sns.FacetGrid(bdat, hue="sex", col="lab_name", col_wrap=4)
# fig.map(sns.lineplot, "session_date", "n_trials", estimator=None, units="subject_nickname").add_legend()
g.set_xticklabels("", rotation=45)
g.savefig(os.path.join(figpath, "trialcounts_malevsfemale.pdf"))
g.savefig(os.path.join(figpath, "trialcounts_malevsfemale.png"))
plt.close("all")

# ================================= #
# WEIGHT VS TRIALCOUNTS
# ================================= #

weight_info = action.Weighing.proj('weight', session_date='DATE(weighing_time)')
weight_matched = (dj.U('subject_uuid', 'session_date') & weight_info) * weight_info
b = (behavior.TrialSet) \
    * (acquisition.Session.proj(session_date='DATE(session_start_time)') & 'session_start_time > "2019-03-01"') \
    * (subject.Subject() & 'subject_birth_date between "2018-09-01" and "2019-02-01"') * subject.SubjectLab() \
    * weight_matched
bdat = pd.DataFrame(b.fetch(order_by='subject_nickname, session_start_time'))

g = sns.lmplot(x="weight", y="n_trials", hue="sex", 
	col="lab_name", units="subject_nickname", 
	col_wrap=4, data=bdat, sharex=False,sharey=False, truncate=True, ci=None)
g.savefig(os.path.join(figpath, "trialcounts_weight_malevsfemale.png"))


