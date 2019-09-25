"""
PsyTrack to get time-varying psychophysical weights for each subject over learning
From: https://github.com/nicholas-roy/psytrack by Nick Roy and Jonathan Pillow
Ported to datajoint by Anne Urai, CSHL, 2019
"""

import pandas as pd
import numpy as np
import sys, os, time
import matplotlib.pyplot as plt
import datajoint as dj
from IPython import embed as shell # for debugging
import datetime

# import wrappers etc
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.analyses import behavior as behavioral_analyses
from dj_tools import *
from ibl_pipeline.analyses import analysis_utils as utils

# install this
from psytrack.hyperOpt import hyperOpt
schema = dj.schema('group_shared_anneurai_analyses')

@schema
class NumStates(dj.Lookup):
    definition = """
    num_states: int      # unique id for spike detection parameter set
    """

@schema
class DateRetrieved(dj.Lookup):
    definition = """
    date_lastsession: int      # unique id for spike detection parameter set
    """


@schema
class DateRetrieved(dj.Lookup):
    definition = """
    date_lastsession: int      # unique id for spike detection parameter set
    """


@schema
class GLMHMMSubject(dj.Computed):
    definition = """
    -> subject.Subject
    -> NumStates
    -> DateRetrieved
    # -> NumTrials
    ---
    glm_weights: longblob #
    transition_matrix: longblob # 
    """

    #key_source = subject.Subject * DateRetrieved * NumStates #overwrite default key_source for make function

    def make(self, key):

        # grab all trials for this subject
        trials = (subject.Subject() & key) * behavior.TrialSet.Trial()
        stim_left, stim_right, resp, feedback = trials.fetch('trial_stim_contrast_left', 
            'trial_stim_contrast_right', 'trial_response_choice', 'trial_feedback_type')

        # # TODO: retrieve days to add dayLength

        # # convert to psytrack format
        # D = {'y':pd.DataFrame(resp)[0].replace({'CCW': 2, 'No Go': np.nan, 'CW': 1}).values}
        # D.update({'inputs':{'contrast_left':stim_left, 'contrast_right':stim_right}})

        # # TODO: specify the weights correctly
        # weights = {'bias' : 1,  # a special key
        #    'contrast_left' : 1,    # use only the first column of s1 from inputs
        #    'contrast_right' : 1}    # use only the first column of s2 from inputs

        # hyper= {'sigInit' : 2**4.,      
        #     'sigma' : [2**-4.]*len(weights),   # Each weight will have it't own sigma, but all initialized the same
        #     'sigDay' : None}        # Not necessary to specify as None, but keeps things general

        # optList = ['sigma']

        # # FIT THE ACTUAL MODEL
        hyp, evd, wMode, hess = hyperOpt(D, hyper, weights, optList)
        
        # TODO: make sure to save the primary key for each trial to link it back up later

        glm_weights, transition_matrix, posterior_probs = run_zoe_func(some_inputs_from_dj) 
        # glm_weights is num_of_states x 3 array; transition matrix is NumStates x NumStates array
        # posterior_probs is array of size NumTrials x NumStates

        glmhmm_result = dict(**key, glm_weights=glm_weights, 
                             transition_matrix=transition_matrix)
        self.insert1(glmhmm_result)
        
        session_start_times, trial_ids = trials.fetch('session_start_time', 'trial_id')
        trials_results = [dict(**key, session_start_time=session_start_time, trial_id=trial_id, posterior_probs=prob) 
                          for prob, session_start_time, trial_id in zip(posterior_probs, session_start_times, trial_ids)]

        self.Trial.insert(trials_results)

        class Trial(dj.Part):
            definition = """
            -> master
            -> behavior.TrialSet.Trial
            ---
            posterior_probs: longblob # of size NumStates
            """


PsyTrack.populate(display_progress=True, limit=10, suppress_errors=True)

