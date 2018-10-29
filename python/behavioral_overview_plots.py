# Anne Urai, CSHL, 2018
# see https://github.com/int-brain-lab/ibllib/tree/master/python/oneibl/examples

import time, re, datetime, os, glob 
from datetime import timedelta
import seaborn as sns 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from IPython import embed as shell

from oneibl.one import ONE
from ibllib.time import isostr2date
from psychofit import psychofit as psy # https://github.com/cortex-lab/psychofit

def get_metadata(mousename):
	
	metadata = {'date_birth': one._alyxClient.get('/weighings?nickname=%s' %mousename),
		'cage': one._alyxClient.get('/cage?nickname=%s' %mousename)}

	return metadata

def get_weights(mousename):
	wei = one._alyxClient.get('/weighings?nickname=%s' %mousename)
	wei = pd.DataFrame(wei)
	wei['date_time'] = pd.to_datetime(wei.date_time)
	wei.sort_values('date_time', inplace=True)
	wei.reset_index(drop=True, inplace=True)
	wei['date'] = wei['date_time'].dt.floor('D')  
	wei['days'] = wei.date - wei.date[0]
	wei['days'] = wei.days.dt.days # convert to number of days from start of the experiment

	return wei

def get_water(mousename):
	wei = one._alyxClient.get('/water-administrations?nickname=%s' %mousename)
	wei = pd.DataFrame(wei)
	wei['date_time'] = pd.to_datetime(wei.date_time)

	# for w in wei:
	# wei['date_time'] = isostr2date(wei['date_time'])
	wei.sort_values('date_time', inplace=True)
	wei.reset_index(drop=True, inplace=True)
	wei['date'] = wei['date_time'].dt.floor('D')  

	wei['days'] = wei.date - wei.date[0]
	wei['days'] = wei.days.dt.days # convert to number of days from start of the experiment

	wei = wei.set_index('date')
	wei.index = pd.to_datetime(wei.index)

	wa_unstacked = wei.pivot_table(index='date', 
		columns='water_type', values='water_administered', aggfunc='sum').reset_index()
	# wa_unstacked = wa_unstacked.set_index('date')
	# wa_unstacked.index = pd.to_datetime(wa_unstacked.index)

	wa_unstacked['date'] = pd.to_datetime(wa_unstacked.date)
	wa_unstacked.set_index('date', inplace=True)

	return wa_unstacked, wei

def fix_date_axis(ax):
	# deal with date axis and make nice looking 
	ax.xaxis_date()
	ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=1))
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
	for item in ax.get_xticklabels():
		item.set_rotation(60)

def get_behavior(mousename, **kwargs):

	# find metadata we need
	eid, details = one.search(subjects=mousename, details=True, **kwargs)

	# sort by date so that the sessions are shown in order
	start_times  = [d['start_time'] for d in details]
	eid 		 = [x for _,x in sorted(zip(start_times, eid))]
	details 	 = [x for _,x in sorted(zip(start_times, details))]

	# grab only behavioral datatypes, all start with _ibl_trials
	types 		= one.list(eid)
	types2 		= [item for sublist in types for item in sublist]
	types2 		= list(set(types2)) # take unique by converting to a set and back to list
	dataset_types = [s for i, s in enumerate(types2) if '_ibl_trials' in s]
	
	# load data over sessions
	for ix, eidx in enumerate(eid):
		dat = one.load(eidx, dataset_types=dataset_types, dclass_output=True)

		# skip if no data, or if there are fewer than 10 trials in this session
		if len(dat.data) == 0:
			continue
		else:
			if len(dat.data[0]) < 10:
				continue
	
		# pull out a dict with variables and their values
		tmpdct = {}
		for vi, var in enumerate(dat.dataset_type):
			k = [item[0] for item in dat.data[vi]]
			tmpdct[re.sub('_ibl_trials.', '', var)] = k

		# add crucial metadata
		tmpdct['subject'] 		= details[ix]['subject']
		tmpdct['users'] 		= details[ix]['users'][0]
		tmpdct['lab'] 			= details[ix]['lab']
		tmpdct['session'] 		= details[ix]['number']
		tmpdct['start_time'] 	= details[ix]['start_time']
		tmpdct['end_time'] 		= details[ix]['end_time']
		tmpdct['trial']         = [i for i in range(len(dat.data[0]))]

		# append all sessions into one dataFrame
		if not 'df' in locals():
			df = pd.DataFrame.from_dict(tmpdct)
		else:
			df = df.append(pd.DataFrame.from_dict(tmpdct), sort=False, ignore_index=True)

	# take care of dates properly
	df['start_time'] = pd.to_datetime(df.start_time)
	df['end_time'] 	 = pd.to_datetime(df.end_time)
	df['date'] 	 	 = df['start_time'].dt.floor("D")

	# convert to number of days from start of the experiment
	df['days'] 		 = df.date - df.date[0]
	df['days'] 		 = df.days.dt.days 

	# add some more handy things
	df['rt'] 		= df['response_times'] - df['stimOn_times']
	df['signedContrast'] = (df['contrastLeft'] - df['contrastRight']) * 100
	df['signedContrast'] = df.signedContrast.astype(int)

	df['correct']   = np.where(np.sign(df['signedContrast']) == df['choice'], 1, 0)
	df.loc[df['signedContrast'] == 0, 'correct'] = np.NaN

	df['choice2'] = df.choice.replace([-1, 0, 1], [0, np.nan, 1]) # code as 0, 100 for percentages
	df['probabilityLeft'] = df.probabilityLeft.round(decimals=2)

	return df

def fit_psychfunc(df):

	# reshape data
	choicedat = df.groupby('signedContrast').agg({'trial':'max', 'choice2':'mean'}).reset_index()
	# print(choicedat)

	# if size(np.abs(df['signedContrast'])) > 6:
	pars, L = psy.mle_fit_psycho(choicedat.values.transpose(), P_model='erf_psycho_2gammas', parstart=np.array([choicedat['signedContrast'].mean(), 20., 0.05, 0.05]), parmin=np.array([choicedat['signedContrast'].min(), 0., 0., 0.]), parmax=np.array([choicedat['signedContrast'].max(), 100., 1, 1]))

	df2 = {'bias':pars[0],'threshold':pars[1], 'lapselow':pars[2], 'lapsehigh':pars[3]}
	return pd.DataFrame(df2, index=[0])

def plot_psychometric(df, ax):

	color = next(ax._get_lines.prop_cycler)['color']
	pars = fit_psychfunc(df)
	xvals = np.linspace(df['signedContrast'].min(), df['signedContrast'].max(), 100)
	tmpdf = pd.DataFrame.from_dict({'xvals': xvals, 
		'yvals': psy.erf_psycho_2gammas(np.transpose(pars.values), xvals)})
	sns.lineplot(x="xvals", y="yvals", color=color, data=tmpdf, ax=ax)

	# datapoints on top
	sns.pointplot(x="signedContrast", y="choice2", color=color, join=False, data=df, ax=ax)
	ax.set(xlabel="Contrast (%)", ylabel="Choose right (%)", ylim=[-0.01,1.01], yticks=[0,0.25,0.5, 0.75, 1])
	ax.grid(True)

def plot_chronometric(df, ax):
	color = next(ax._get_lines.prop_cycler)['color']
	sns.pointplot(x="signedContrast", y="rt", color=color, estimator=np.median, join=True, data=df, ax=ax)
	ax.set(xlabel="Contrast (%)", ylabel="RT (s)")
	ax.grid(True)

# ============================================= #
# START BIG OVERVIEW PLOT
# ============================================= #

## INITIALIZE A FEW THINGS
sns.set()
sns.set_context(context="paper")
current_palette = sns.color_palette()
# set a new palette for biased blocks: black, purple, orange
one = ONE() # initialize

# get a list of all mice that are currently training
subjects 	= pd.DataFrame(one._alyxClient.get('/subjects?alive=True'))

for i, mouse in enumerate(subjects['nickname']):

	try:

		# MAKE THE FIGURE, divide subplots using gridspec
		print(mouse)
		wa_unstacked, wa 	= get_water(mouse)
		wei 	= get_weights(mouse)
		behav 	= get_behavior(mouse)

		fig, axes = plt.subplots(ncols=5, nrows=4, constrained_layout=False,
	        gridspec_kw=dict(width_ratios=[2,2,1,1,1], height_ratios=[1,1,1,1]), figsize=(11.69, 8.27))

		# ============================================= #
		# GENERAL METADATA
		# ============================================= #

		# write mouse info
		ax = axes[0,0]
		ax.annotate('Mouse %s (%s), DoB %s \n user %s, %s\n strain %s, cage %s \n %s' %(subjects['nickname'][i],
		 subjects['sex'][i], subjects['birth_date'][i], 
		 subjects['responsible_user'][i], subjects['lab'][i],
		 subjects['strain'][i], subjects['litter'][i], subjects['description'][i]),
		 xy=(0.5, 0.5), xycoords='axes fraction', va='center', ha='center')
		ax.axis('off')

		# ============================================= #
		# WEIGHT CURVE AND WATER INTAKE
		# ============================================= #

		# weight on top
		ax = axes[1,0]
		sns.lineplot(x="date_time", y="weight", color="black", markers=True, data=wei, ax=ax)
		sns.scatterplot(x="date_time", y="weight", color="black", data=wei, ax=ax)
		ax.set(xlabel='', ylabel="Weight (g)", 
			xlim=[wei.date_time.min()-timedelta(days=1), wei.date_time.max()+timedelta(days=1)])
		fix_date_axis(ax)

		# use pandas plot for a stacked bar - water types
		ax = axes[2,0]
		sns.set_palette("colorblind") # palette for water
		wa_unstacked.loc[:,['Water','Hydrogel']].plot.bar(stacked=True, ax=ax)
		l = ax.legend()
		l.set_title('')
		ax.set(ylabel="Water intake (mL)", xlabel='')

		# fix dates, known to be an issue in pandas/matplotlib
		ax.set_xticklabels([dt.strftime('%b-%d') if dt.weekday() is 1 else "" for dt in wa_unstacked.index.to_pydatetime()])
		for item in ax.get_xticklabels():
			item.set_rotation(60)

		# ============================================= #
		# TRIAL COUNTS AND PERFORMANCE
		# ============================================= #

		# performance on easy trials
		ax = axes[3,0]
		behav['correct_easy'] = behav.correct
		behav.loc[np.abs(behav['signedContrast']) < 50, 'correct_easy'] = np.NaN
		correct_easy = behav.groupby(['start_time'])['correct_easy'].mean().reset_index()
		
		sns.lineplot(x="start_time", y="correct_easy", markers=True, color="black", data=correct_easy, ax=ax)
		sns.scatterplot(x="start_time", y="correct_easy", color="black", data=correct_easy, ax=ax)
		ax.set(xlabel='', ylabel="Performance on easy trials", 
			xlim=[behav.date.min()-timedelta(days=1), behav.date.max()+timedelta(days=1)],
			yticks=[0.5, 0.75, 1], ylim=[0.4, 1.01])
		ax.yaxis.label.set_color("black")

		# overlay trial counts
		trialcounts = behav.groupby(['start_time'])['trial'].max().reset_index()
		righty = ax.twinx()
		sns.lineplot(x="start_time", y="trial", markers=True, color="firebrick", data=trialcounts, ax=righty)
		sns.scatterplot(x="start_time", y="trial", color="firebrick", data=trialcounts, ax=righty)
		righty.yaxis.label.set_color("firebrick")
		righty.grid(False)
		fix_date_axis(righty)
		fix_date_axis(ax)
		righty.set(xlabel='', ylabel="Trial count", 
			xlim=[behav.date.min()-timedelta(days=1), behav.date.max()+timedelta(days=1)])

		# ============================================= #
		# PSYCHOMETRIC FUNCTION FITS OVER TIME
		# ============================================= #

		# fit psychfunc on choice fraction, rather than identity
		pars = behav.groupby(['start_time', 'probabilityLeft']).apply(fit_psychfunc).reset_index()
		parsdict = {'threshold': r'Threshold $(\sigma)$', 'bias': r'Bias $(\mu)$', 
			'lapselow': r'Lapse low $(\gamma)$', 'lapsehigh': r'Lapse high $(\lambda)$'}

		# pick a good-looking diverging colormap with black in the middle
		cmap = sns.diverging_palette(220, 20, n=len(behav['probabilityLeft'].unique()), center="dark")
		if len(behav['probabilityLeft'].unique()) == 1:
			cmap = "gist_gray"
		sns.set_palette(cmap)

		for pidx, (var, labelname) in enumerate(parsdict.items()):
			ax = axes[pidx,1]
			sns.lineplot(x="start_time", y=var, hue="probabilityLeft", palette=cmap, data=pars, legend=None, ax=ax)
			sns.scatterplot(x="start_time", y=var, hue="probabilityLeft", palette=cmap, data=pars, legend=None, ax=ax)
			ax.set(xlabel='', ylabel=labelname, xlim=[behav.date.min()-timedelta(days=1), behav.date.max()+timedelta(days=1)])

			fix_date_axis(ax)
			if pidx == 0:
				ax.set(title=r'$\gamma + (1 -\gamma-\lambda)  (erf(\frac{x-\mu}{\sigma} + 1)/2$')
			if pidx < 3:
				ax.set(xticklabels=[])

		# ============================================= #
		# LAST THREE SESSIONS
		# ============================================= #

		didx = 1
		for day in behav['days'].unique():
			if day < behav['days'].max()-3:
				continue

			dat = behav.loc[behav['days'] == day, :]
			didx += 1

			# PSYCHOMETRIC FUNCTION
			ax = axes[0, didx]
			dat.groupby(['probabilityLeft']).apply(plot_psychometric, (ax))
			ax.set(title=pd.to_datetime(dat['start_time'].unique()[0]).strftime('%b-%d, %A'))

			# CHRONOMETRIC FUNCTION
			ax = axes[1, didx]
			dat.groupby(['probabilityLeft']).apply(plot_chronometric, (ax))

			# RTS THROUGHOUT SESSION - add running mean
			# def rolling_mean(data):
			#     return pd.rolling_mean(data, 10).mean()

			ax = axes[2, didx]
			sns.scatterplot(x='trial', y='rt', hue='correct', 
				palette={1:"forestgreen", 0:"crimson"},
				alpha=.5, data=dat, ax=ax, legend=False)
			sns.lineplot(x='trial', y='rt', color='black', ci=None, 
				data=dat[['trial', 'rt']].rolling(10).median()) 
			ax.set(xlabel="Trial number", ylabel="RT (s)")
			ax.set_yscale("log")
			ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
			# ax.set_xticks(np.arange(0, dat['trial'].max(), 100))

		for i in range(3):
			axes[i,3].set(ylabel='')
			axes[i,4].set(ylabel='')
		righty.set(ylabel='Trial count')

		plt.tight_layout()
		fig.savefig('/Users/urai/Google Drive/Rig building WG/DataFigures/BehaviourData_Weekly/AlyxPlots/%s_overview.pdf' %mouse)
	except:
		pass

	
