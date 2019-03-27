﻿from oneibl.one import ONE
import pandas as pd
import numpy as np
from os import listdir, getcwd
from os.path import isfile, join
import re
from IPython import embed as shell

one = ONE() # initialize
# one = ONE(base_url='https://dev.alyx.internationalbrainlab.org')

# ==================== #
# ONE/ALYX
# ==================== #

def get_weights(mousename):

    wei = one.alyx.get('/weighings?nickname=%s' %mousename)
    wei = pd.DataFrame(wei)

    if not wei.empty:
        wei['date_time'] = pd.to_datetime(wei.date_time)
        wei.sort_values('date_time', inplace=True)
        wei.reset_index(drop=True, inplace=True)
        wei['date'] = wei['date_time'].dt.floor('D')  
        wei['days'] = wei.date - wei.date[0]
        wei['days'] = wei.days.dt.days # convert to number of days from start of the experiment

    return wei

def get_water(mousename):
    
    wei = one.alyx.get('/water-administrations?nickname=%s' %mousename)
    wei = pd.DataFrame(wei)

    if not wei.empty:
        wei['date_time'] = pd.to_datetime(wei.date_time)
        wei.sort_values('date_time', inplace=True)
        wei.reset_index(drop=True, inplace=True)
        wei['date'] = wei['date_time'].dt.floor('D')  

        wei['days'] = wei.date - wei.date[0]
        wei['days'] = wei.days.dt.days # convert to number of days from start of the experiment

    return wei

def get_water_weight(mousename):

    wei = get_weights(mousename)
    wa  = get_water(mousename)

    if not (wei.empty or wa.empty):
        wei = wei.groupby(['date']).mean().reset_index()
        wa.reset_index(inplace=True)

        # make sure that NaNs are entered for days with only water or weight but not both
        combined = pd.merge(wei, wa, on="date", how='outer')
        combined = combined[['date', 'weight', 'water_administered', 'water_type']]

        # also grab the info about water restriction
        restr = one.alyx.get('/subjects/%s' %mousename)

        # only if the mouse is on water restriction, add its baseline weight
        if restr['last_water_restriction']:

            # TODO: add list of water restrictions with start and end dates
            # see: https://int-brain-lab.slack.com/archives/CBW27C8D7/p1550266382011300
            # https://github.com/int-brain-lab/ibllib/issues/52

            baseline = pd.DataFrame.from_dict({'date': pd.to_datetime(restr['last_water_restriction']), 
                'weight': restr['reference_weight'], 'index':[0]})

            # add the baseline to the combined df
            combined = combined.append(baseline, sort=False)

        else:
            baseline = pd.DataFrame.from_dict({'date': None, 'weight': restr['reference_weight'], 'index':[0]})

        combined = combined.sort_values(by='date')
        combined['date'] = combined['date'].dt.floor("D") # round the time of the baseline weight down to the day

        combined = combined.reset_index()
        combined = combined.drop(columns='index')

        # also indicate all the dates as days from the start of water restriction (for easier plotting)
        combined['days'] = combined.date - combined.date[0]
        combined['days'] = combined.days.dt.days # convert to number of days from start of the experiment

    else:
        combined     = pd.DataFrame()
        baseline     = pd.DataFrame()

    return combined, baseline

def get_behavior(mousename, **kwargs):

    # find metadata we need
    eid, details = one.search(subjects=mousename, details=True, **kwargs)

    # sort by date so that the sessions are shown in order
    start_times  = [d['start_time'] for d in details]
    eid          = [x for _,x in sorted(zip(start_times, eid))]
    details      = [x for _,x in sorted(zip(start_times, details))]

    # load data over sessions
    for ix, eidx in enumerate(eid):
        dat = one.load(eidx, dataset_types=['_iblrig_taskData.raw'], dclass_output=True)

        # skip if no data, or if there are fewer than 10 trials in this session
        if len(dat.data) == 0:
            continue
        else:
            try:
                if len(dat.data[0]) < 10:
                    continue
            except:
                continue

        # pull out a dict with variables and their values
        tmpdct = {}
        for vi, var in enumerate(dat.dataset_type):
            if dat.data[vi].ndim == 1:
                tmpdct[re.sub('_ibl_trials.', '', var)] = dat.data[vi]
            elif dat.data[vi].ndim == 2: # intervals
                if dat.data[vi].shape[1] == 1:
                    tmpdct[re.sub('_ibl_trials.', '', var)] = [item[0] for item in dat.data[vi]]
                elif dat.data[vi].shape[1] == 1:
                    tmpdct[re.sub('_ibl_trials.', '', var) + '_start'] = dat.data[vi][:,0]
                    tmpdct[re.sub('_ibl_trials.', '', var) + '_end']   = dat.data[vi][:,1]
            else:
                print('behavioral data %s has more than 2 dimensions, not sure what it is'%var)
                shell()

        # ADD SOME CRUCIAL THINGS
        if 'probabilityLeft' not in tmpdct.keys():
            tmpdct['probabilityLeft'] = 0.5 * np.ones(tmpdct['choice'].shape)
        if 'included' not in tmpdct.keys():
            tmpdct['included'] = np.ones(tmpdct['choice'].shape)
        if 'goCue_times' not in tmpdct.keys():
            try:
                tmpdct['goCue_times'] = tmpdct['stimOn_times']
            except:
                tmpdct['goCue_times'] = np.full(tmpdct['choice'].shape, np.nan)

        # add crucial metadata
        tmpdct['subject']       = details[ix]['subject']
        tmpdct['users']         = details[ix]['users'][0]
        tmpdct['lab']           = details[ix]['lab']
        tmpdct['session']       = details[ix]['number']
        tmpdct['start_time']    = details[ix]['start_time']
        tmpdct['end_time']      = details[ix]['end_time']
        tmpdct['trial']         = [i for i in range(len(dat.data[0]))]
        tmpdct['task_protocol'] = details[ix]['task_protocol']

        # append all sessions into one dataFrame
        if not 'df' in locals():
            df = pd.DataFrame.from_dict(tmpdct)
        else:
            df = df.append(pd.DataFrame.from_dict(tmpdct), sort=False, ignore_index=True)

    # POST-PROCESSING
    # take care of dates properly
    df['start_time'] = pd.to_datetime(df.start_time)
    df['end_time']   = pd.to_datetime(df.end_time)
    df['date']       = df['start_time'].dt.floor("D")

    # convert to number of days from start of the experiment
    df['days']       = df.date - df.date[0]
    df['days']       = df.days.dt.days 

    # fix a bug
    df['contrastLeft'] = np.abs(df['contrastLeft'])
    df['contrastRight'] = np.abs(df['contrastRight'])

    # make sure there are no NA values in the contrasts
    df.fillna({'contrastLeft': 0, 'contrastRight': 0}, inplace=True)
    df['signedContrast'] = (- df['contrastLeft'] + df['contrastRight']) * 100
    df['signedContrast'] = df.signedContrast.astype(int)

    # flip around choice coding - go from wheel movement to percept
    df['choice'] = -1*df['choice']
    df['correct']   = np.where(np.sign(df['signedContrast']) == df['choice'], 1, 0)
    df.loc[df['signedContrast'] == 0, 'correct'] = np.NaN
    df['choice2'] = df.choice.replace([-1, 0, 1], [0, np.nan, 1]) # code as 0, 100 for percentages

    # add some more handy things
    df['rt']        = df['response_times'] - df['goCue_times']
    df.loc[df.choice == 0, 'rt'] = np.nan # don't count RT if there was no response

    # for trainingChoiceWorld, make sure all probabilityLeft = 0.5
    df['probabilityLeft'] = df.probabilityLeft.round(decimals=2)
    df['probabilityLeft_block'] = df['probabilityLeft']
    df.fillna({'task_protocol':'unknown'}, inplace=True)
    df.loc[df['task_protocol'].str.contains("trainingChoiceWorld"), 'probabilityLeft_block'] = 0.5

    return df
 