from oneibl.one import ONE
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

    # wei = wei.set_index('date')
    # wei.index = pd.to_datetime(wei.index)

    wa_unstacked = wei.pivot_table(index='date', 
        columns='water_type', values='water_administered', aggfunc='sum').reset_index()
    # wa_unstacked = wa_unstacked.set_index('date')
    # wa_unstacked.index = pd.to_datetime(wa_unstacked.index)

    wa_unstacked['date'] = pd.to_datetime(wa_unstacked.date)
    wa_unstacked.set_index('date', inplace=True)

    return wa_unstacked, wei


def get_water_weight(mousename):

    wei = get_weights(mousename)
    wa_unstacked, wa = get_water(mousename)
    wa.reset_index(inplace=True)

    # also grab the info about water restriction
    # shell()
    restr = mouse_data_ = one._alyxClient.get('/subjects/%s' %mousename)

    # make sure that NaNs are entered for days with only water or weight but not both
    combined = pd.merge(wei, wa, on="date", how='outer')
    combined = combined[['date', 'weight', 'water_administered', 'water_type']]

    # if no hydrogel was ever given to this mouse, add it anyway with NaN

    # remove those weights below current water restriction start
    combined = combined[combined.date >= pd.to_datetime(restr['last_water_restriction'])]

    # add a weight measurement on day 0 that shows the baseline weight
    combined = combined.append(pd.DataFrame.from_dict({'date': pd.to_datetime(restr['last_water_restriction']), 
        'weight': restr['reference_weight'], 'water_administered': np.nan, 'water_type': np.nan, 'index':[0]}), 
        sort=False)
    combined = combined.sort_values(by='date')
    combined['date'] = combined['date'].dt.floor("D") # round the time of the baseline weight down to the day
    
    combined = combined.reset_index()
    combined = combined.drop(columns='index')

    # also indicate all the dates as days from the start of water restriction (for easier plotting)
    combined['days'] = combined.date - combined.date[0]
    combined['days'] = combined.days.dt.days # convert to number of days from start of the experiment
    

    return combined



def get_behavior(mousename, **kwargs):

    # find metadata we need
    eid, details = one.search(subjects=mousename, details=True, **kwargs)

    # sort by date so that the sessions are shown in order
    start_times  = [d['start_time'] for d in details]
    eid          = [x for _,x in sorted(zip(start_times, eid))]
    details      = [x for _,x in sorted(zip(start_times, details))]

    # grab only behavioral datatypes, all start with _ibl_trials
    types       = one.list(eid)
    types2      = [item for sublist in types for item in sublist]
    types2      = list(set(types2)) # take unique by converting to a set and back to list
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
        tmpdct['subject']       = details[ix]['subject']
        tmpdct['users']         = details[ix]['users'][0]
        tmpdct['lab']           = details[ix]['lab']
        tmpdct['session']       = details[ix]['number']
        tmpdct['start_time']    = details[ix]['start_time']
        tmpdct['end_time']      = details[ix]['end_time']
        tmpdct['trial']         = [i for i in range(len(dat.data[0]))]

        # append all sessions into one dataFrame
        if not 'df' in locals():
            df = pd.DataFrame.from_dict(tmpdct)
        else:
            df = df.append(pd.DataFrame.from_dict(tmpdct), sort=False, ignore_index=True)

    # take care of dates properly
    df['start_time'] = pd.to_datetime(df.start_time)
    df['end_time']   = pd.to_datetime(df.end_time)
    df['date']       = df['start_time'].dt.floor("D")

    # convert to number of days from start of the experiment
    df['days']       = df.date - df.date[0]
    df['days']       = df.days.dt.days 

    # add some more handy things
    df['rt']        = df['response_times'] - df['stimOn_times']
    df['signedContrast'] = (df['contrastLeft'] - df['contrastRight']) * 100
    df['signedContrast'] = df.signedContrast.astype(int)

    df['correct']   = np.where(np.sign(df['signedContrast']) == df['choice'], 1, 0)
    df.loc[df['signedContrast'] == 0, 'correct'] = np.NaN

    df['choice2'] = df.choice.replace([-1, 0, 1], [0, np.nan, 1]) # code as 0, 100 for percentages
    df['probabilityLeft'] = df.probabilityLeft.round(decimals=2)

    return df
 