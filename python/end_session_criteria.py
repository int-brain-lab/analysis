'''
Plot various criteria for each mouse session
'''

import pandas as pd
import numpy as np
import sys, os
import datajoint as dj

sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'Documents/Python Scripts'))
sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'Documents/Github/analysis/python'))

from ibl_pipeline import acquisition, behavior

schema = dj.schema('user_mileswells_end_criteria')

@schema
class EndCriteria(dj.Lookup):
    definition = """
    end_criteria: varchar(32)
    """
    contents = zip(['long_rt',
                    'perf<40',
                    'perf_ez<40',
                    '<400_trials',
                    '>45_min_&_stopped',
                    '>90_min'
                    ])

@schema
class SessionEndCriteria(dj.Computed):
    definition = """
    -> acquisition.Session
    ---
    end_status:         varchar(32) # First end status to be triggered
    end_status_index:   int # trial_id index when status first triggered 
    """

    key_source = behavior.CompleteTrialSession

    def make(self, key):

        trials = behavior.TrialSet.Trial & key
        trials = trials.proj(
            'trial_response_choice',
            'trial_response_choice',
            'trial_response_time',
            'trial_stim_on_time',
            'trial_start_time',
            signed_contrast='trial_stim_contrast_right \
                - trial_stim_contrast_left',
            rt='trial_response_time - trial_stim_on_time',
            correct='trial_feedback_type = 1')
        behav = pd.DataFrame(trials.fetch(order_by='trial_id'))

        if behav.empty:
            return
        # task_protocol = (acquisition.Session & key).fetch1('task_protocol')  # TODO

        ## CALCULATE CRITERIA
        rt_win_size = 20  # Size of reaction time rolling window
        perf_win_size = 50  # Size of performance rolling window
        min_trials = 400  # Minimum number of trials for criteria to apply

        behav['correct_easy'] = behav.correct
        behav.loc[np.abs(behav['signed_contrast']) < .5, 'correct_easy'] = np.NaN
        behav['n_trials_last_5'] = behav['trial_start_time'].expanding().apply(
            lambda x: sum((x[-1] - x[0:-1]) < 5 * 60), raw=True)

        # Local and session median reaction times
        behav['RT_local'] = behav['rt'].rolling(rt_win_size).median()
        behav['RT_global'] = behav['rt'].expanding().median()
        behav['RT_delta'] = behav['RT_local'] > (behav['RT_global'] * 5)

        # Local and global performance
        behav['perf_local'] = behav['correct'].rolling(perf_win_size).apply(lambda x: sum(x) / x.size, raw=True)
        behav['perf_global'] = behav['correct'].expanding().apply(lambda x: sum(x) / x.size, raw=True)
        behav['perf_delta'] = (behav['perf_global'] - behav['perf_local']) / behav['perf_global']

        # Performance for easy trials only
        last = lambda x: np.nonzero(~np.isnan(x))[0][:-perf_win_size]  # Find last n values that aren't nan
        behav['perf_local_ez'] = (behav['correct_easy'].expanding()
                                  .apply(lambda x: sum(last(x)) / last(x).size if last(x).size else np.nan, raw=True))
        behav['perf_global_ez'] = behav['correct_easy'].expanding().apply(lambda x: (sum(x == 1) / sum(~np.isnan(x))), raw=True)
        behav['perf_delta_ez'] = (behav['perf_global_ez'] - behav['perf_local_ez']) / behav['perf_global_ez']

        status_idx = dict.fromkeys(EndCriteria.contents)
        status_idx['long_rt'] = ((behav.RT_delta) & (behav.index > min_trials)).idxmax() if (
                    (behav.RT_delta) & (behav.index > min_trials)).any() else np.nan
        status_idx['perf_ez<40'] = ((behav['perf_delta_ez'] > 0.4) & (behav.index > min_trials)).idxmax()
        status_idx['perf<40'] = ((behav['perf_delta_ez'] > 0.4) & (behav.index > min_trials)).idxmax()
        status_idx['<400_trials'] = ((behav.trial_start_time > 45 * 60) & (behav.index < min_trials)).idxmax()
        status_idx['>45_min_&_stopped'] = (
                    (behav.trial_start_time > 45 * 60) & (behav['n_trials_last_5'] < 45)).idxmax()
        status_idx['>90_min'] = (behav.trial_start_time > 90 * 60).idxmax()
        status_idx = {k: v for (k, v) in status_idx.items() if v > 0}

        if status_idx:
            criterion = min(status_idx, key=status_idx.get)
            key['end_status'] = criterion
            key['end_status_index'] = status_idx[criterion]
            self.insert1(key)
