#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:54 2021

This module contains functions to report on histology metrics on insertions
from Alyx.

@author: sjwest
"""
from one.api import ONE
from pathlib import Path
import requests
from one import params


def check_insertions(alyx_project = 'ibl_neuropixel_brainwide_01'):
    """Check how many insertions recorded and histology reoncstructed
    
    And return number of insertions that are recorded and have no histology.
    
    Parameters
    ----------
    alyx_project : TYPE, optional
        Which project should Alyx queries be limited to?  Use 
        one.alyx.rest('projects', 'list') to see a list of possible projects. 
        The default is 'ibl_neuropixel_brainwide_01'.

    Returns
    -------
    sub_dict : dict
        Returned dict contains the number of insertions with recordings, number
        of insertions with histology, and nubmer of insertions with recordings
        that still need histology.

    """
    one = ONE()
    traj_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track', project=alyx_project)
    
    traj_micro = one.alyx.rest('trajectories', 'list', provenance='Micro-manipulator', project=alyx_project)
    
    ins_hist = [sess['probe_insertion'] for sess in traj_hist]
    ins_micro = [sess['probe_insertion'] for sess in traj_micro]
    
    # create dict to return
    ins_dict = dict()
    # compute numbers
    ins_dict['insertions_histology_number'] = len(ins_hist)
    # 858
    ins_dict['insertions_micro-manipulator_number'] = len(ins_micro)
    # 944
    
    # compute number of insertions that do NOT have histology
    ins_micro_no_hist = [x for x in ins_micro if x not in set(ins_hist)]
    ins_dict['insertions_micro-manipulator_no_histology_number'] = len(ins_micro_no_hist)
    # 103
    return ins_dict


