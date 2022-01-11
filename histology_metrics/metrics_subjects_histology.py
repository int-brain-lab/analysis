#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:54 2021

This module contains functions to report on histology metrics on subjects
from Alyx.

@author: sjwest
"""
from one.api import ONE
from pathlib import Path
import requests
from one import params


def check_subjects_numbers_need_histology_imaging(alyx_project = 'ibl_neuropixel_brainwide_01'):
    """Check subjects need histology IMAGING
    
    These are subjects that have PLANNED trajectories but no HISTOLOGY TRACK
    trajectories.
    
    Parameters
    ----------
    alyx_project : TYPE, optional
        Which project should Alyx queries be limited to?  Use 
        one.alyx.rest('projects', 'list') to see a list of possible projects. 
        The default is 'ibl_neuropixel_brainwide_01'.

    Returns
    -------
    sub_dict : dict
        Returned dict contains the number of subjects with planned trajectories,
        the number of subjects with histology reconstructed trajectories, and
        the a set of subject IDs that have planned trajectories but no histology
        reconstructed trajectories.

    """
    one = ONE()
    
    # get all trajectories : Planned, Histology
    traj_planned = one.alyx.rest('trajectories', 'list', provenance='Planned', project=alyx_project)
    traj_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track', project=alyx_project)
    
    # get all subjects from these trajectories
    subjP = [sess['session']['subject'] for sess in traj_planned]
    subj_planned = list(set(subjP)) # only unique subjects
    subjH = [sess['session']['subject'] for sess in traj_hist]
    subj_hist = list(set(subjH))
    
    # create dict to return
    sub_dict = dict()
    
    # compute numbers
    sub_dict['number_subjects_planned'] = len(subj_planned)
    # currently 108
    
    sub_dict['number_subjects_histology'] = len(subj_hist)
    # currently 108
    
    #sub_dict[''] = set(subj_hist) - set(subj_planned)
    # currently {'DY_006', 'NYU-14', 'NYU-39'}
    
    sub_dict['subjects_planned_no_histology'] = set(subj_planned) - set(subj_hist)
    # currently {'CSH_ZAD_028', 'UCLA013', 'UCLA014'}
    
    return sub_dict



def check_histology_exists(subject, lab):
    """Check if histology Exists for a subject and lab
    
    Check Flatiron for registered histology data.
    
    Parameters
    ----------
    subject : str
        The subject ID. eg. 'KS052'.
    lab : str
        The lab ID. eg. 'cortexlab'.

    Returns
    -------
    bool
        Boolean to indicate whether the histology exists.

    """

    if lab == 'hoferlab':
        lab_temp = 'mrsicflogellab'
    elif lab == 'churchlandlab_ucla':
        lab_temp = 'churchlandlab'
    else:
        lab_temp = lab

    par = params.get()

    FLAT_IRON_HIST_REL_PATH = Path('histology', lab_temp, subject,
                                   'downsampledStacks_25', 'sample2ARA')
    baseurl = (par.HTTP_DATA_SERVER + '/' + '/'.join(FLAT_IRON_HIST_REL_PATH.parts))
    r = requests.head(baseurl, auth=(par.HTTP_DATA_SERVER_LOGIN, par.HTTP_DATA_SERVER_PWD))
    if 'Location' in r.headers:
        return True
    else:
        return False



def check_subjects_labs_need_histology_reconstruction(alyx_project = 'ibl_neuropixel_brainwide_01'):
    """Check subjects & labs need histology RECONSTRUCTION
    
    These are subjects/lab numbers that have MICRO-MANIPULATOR trajectories but
    do not have HISTOLOGY TRACK trajectories.
    

    Parameters
    ----------
    alyx_project : TYPE, optional
        Which project should Alyx queries be limited to?  Use 
        one.alyx.rest('projects', 'list') to see a list of possible projects. 
        The default is 'ibl_neuropixel_brainwide_01'.

    Returns
    -------
    sub_dict : dict
        Returned dict contains the set of subject IDs that need histology,
        and a dict of the lab counts that need histology.

    """
    
    one = ONE()
    traj_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track', project=alyx_project)
    
    traj_micro = one.alyx.rest('trajectories', 'list', provenance='Micro-manipulator', project=alyx_project)
    
    ins_hist = [sess['probe_insertion'] for sess in traj_hist]
    ins_micro = [sess['probe_insertion'] for sess in traj_micro]
    
    # compute numbers
    #len(ins_hist)
    # 858
    #len(ins_micro)
    # 944
    
    # compute number of insertions that do NOT have histology
    ins_micro_no_hist = [x for x in ins_micro if x not in set(ins_hist)]
    #len(ins_micro_no_hist)
    # 103
    
    # create dict to return
    sub_lab_dict = dict()
    
    # compute subjects for insertions that need tracing
    subj_micro_no_hist = [sess['session']['subject'] for sess in traj_micro if sess['probe_insertion'] in set(ins_micro_no_hist)]
    
    subj_micro_no_hist_set = set(subj_micro_no_hist)
    #len(subj_micro_no_hist_set)
    # 33 subjects
    sub_lab_dict['subjects_need_histology'] = subj_micro_no_hist_set
    
    # compute labs for insertions that need tracing
    subjs = one.alyx.rest('subjects', 'list', project=alyx_project)
    labs_micro = list()
    for s in subj_micro_no_hist_set:
        for m in subjs:
            if m['nickname'] == s:
                labs_micro.append(m['lab'])
    
    labs_micro_dict = {i:labs_micro.count(i) for i in labs_micro}
    
    
    sub_lab_dict['labs_need_histology'] = labs_micro_dict
    
    return sub_lab_dict

