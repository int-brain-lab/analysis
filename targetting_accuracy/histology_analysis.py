#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:49:13 2020

@author: ibladmin
"""
import pandas as pd
from ibllib import atlas
from oneibl.one import ONE
from ibllib.pipes import histology
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from ibllib.ephys.neuropixel import SITES_COORDINATES
from scipy.spatial import distance
# Function definitions

def planed_vs_histology(trj_hist):
    '''
    Measures the error between the planned probe insertion and the one recorded
    in the manipulator on recording day.
    INPUT:
        trj_hist: Histology dataframe
    OUTPUT:
        figure

    '''
    trj_hist['mean_cecler'] = np.nan
    trj_hist['r_miss'] = np.nan
    trj_hist['r_out'] = np.nan
    
    for i in range(len(trj_hist)):
        pen = trj_hist.iloc[i,:]
        hist_probe = {'x': pen['x'],
                      'y':  pen['y'],
                      'z':  pen['z'],
                      'phi':  pen['phi'],
                      'theta': pen['theta'],
                      'depth':  pen['depth']}
        pln_probe ={'x': pen['pln_x'],
                      'y':  pen['pln_y'],
                      'z':  pen['pln_z'],
                      'phi': pen['pln_phi'],
                      'theta': pen['pln_theta'],
                      'depth': pen['pln_depth']}
        try:
            trj_hist.iloc[i,-3], trj_hist.iloc[i,-2], \
            trj_hist.iloc[i,-1] = compare_2_probes(pln_probe, hist_probe)
        except:
            pass
    
    
    figure, delta = plt.subplots(1,2, figsize =[10,5])
    plt.sca(delta[0])
    plt.xticks(rotation = 45)
    plt.rcParams.update({'font.size': 12})
    sns.swarmplot(data = trj_hist, y = 'mean_cecler',
                 x = 'lab', ax = delta[0], color='k')
    sns.boxplot(data = trj_hist, y = 'mean_cecler',
                 x = 'lab', ax = delta[0], color='gray')
    delta[0].set_xlabel('Laboratory Location')
    delta[0].spines['top'].set_visible(False)
    delta[0].spines['right'].set_visible(False)
    delta[0].set_ylabel('Mean euclidean error (um)')
    plt.sca(delta[1])
    plt.xticks(rotation = 45)
    plt.rcParams.update({'font.size': 12})
    sns.swarmplot(data = trj_hist, y = 'r_out',
                 x = 'lab', ax = delta[1], color='k')
    sns.boxplot(data = trj_hist, y = 'r_out',
                 x = 'lab', ax = delta[1], color='gray')
    delta[1].set_xlabel('Laboratory Location')
    delta[1].spines['top'].set_visible(False)
    delta[1].spines['right'].set_visible(False)
    delta[1].set_ylabel('Fraction of regions missed')
    
    return delta

def compare_2_probes(dict1,dict2):
    '''
    Given two probes in the following format, it calculates the cerror between 
    the probes. Dict2 is compared to dict1
    Dict format for Insertion function
    {'x': 544.0,
    'y': 1285.0,
    'z': 0.0,
    'phi': 0.0,
    'theta': 5.0,
    'depth': 4501.0}
    INPUT:
        dict_1,dict_2: 2 dictionaries with probe coordinates
    OUTPUT:
        cecler: Comulative ecludian error
        r_miss: each channel that debiates from the intetion adds 1/divided by
        total number of channels. 1 is all are different
        r_out: number of unique regions in dict1 missed divided by total possible
    '''
    # mean eucledian error
    dict_1_ins = atlas.Insertion.from_dict(dict1)
    dict1_regions, _, dict1_xyz = histology.get_brain_regions(dict_1_ins.xyz)
    dict_2_ins = atlas.Insertion.from_dict(dict2)
    dict2_regions, _, dict2_xyz = histology.get_brain_regions(dict_2_ins.xyz)
    cecler = 0
    for i  in range(len(dict1_xyz)):
        cecler += distance.euclidean(dict1_xyz[i,:], dict2_xyz[i,:])
    mean_cecler = (cecler/len(dict1_xyz))*(10**6)
    
    #score areas missmatch
    r_miss = 0
    for i in range(len(dict1_regions['id'])):
        r_miss += 1 - int(dict1_regions['id'][i] == dict2_regions['id'][i])
    r_miss = r_miss/len(dict1_regions['id'])
    
    #percentage of regions totally missed
    r_out = (1 - len(np.intersect1d(np.unique(dict1_regions['id']), 
                               np.unique(dict2_regions['id'])))/
             len(np.unique(dict1_regions['id'])))
    
    return mean_cecler, r_miss, r_out


def manipulator_vs_histology(trj_hist):
    '''
    Measures the error between the manipulator or planned probe insertion 
    and the one from histology
    INPUT:
        trj_theory: Planned trajectory
        trj_hist: Histology trajectory
    OUTPUT:
        trj_hist: Histology dataframe with extra 3 columns
        mean_cecler, r_miss, r_out (see def compare 2 probes for definition)
    '''
    
    trj_hist['mean_cecler'] = np.nan
    trj_hist['r_miss'] = np.nan
    trj_hist['r_out'] = np.nan
    
    for i in range(len(trj_hist)):
        pen = trj_hist.iloc[i,:]
        hist_probe = {'x': pen['x'],
                      'y':  pen['y'],
                      'z':  pen['z'],
                      'phi':  pen['phi'],
                      'theta': pen['theta'],
                      'depth':  pen['depth']}
        micro_probe ={'x': pen['mcr_x'],
                      'y':  pen['mcr_y'],
                      'z':  pen['mcr_z'],
                      'phi': pen['mcr_phi'],
                      'theta': pen['mcr_theta'],
                      'depth': pen['mcr_depth']}
        try:
            trj_hist.iloc[i,-3], trj_hist.iloc[i,-2], \
            trj_hist.iloc[i,-1] = compare_2_probes(micro_probe, hist_probe)
        except:
            pass
    
    
    figure, delta = plt.subplots(1,2, figsize =[10,5])
    plt.sca(delta[0])
    plt.xticks(rotation = 45)
    plt.rcParams.update({'font.size': 12})
    sns.swarmplot(data = trj_hist, y = 'mean_cecler',
                 x = 'lab', ax = delta[0], color='k')
    sns.boxplot(data = trj_hist, y = 'mean_cecler',
                 x = 'lab', ax = delta[0], color='gray')
    delta[0].set_xlabel('Laboratory Location')
    delta[0].spines['top'].set_visible(False)
    delta[0].spines['right'].set_visible(False)
    delta[0].set_ylabel('Mean euclidean error (um)')
    plt.sca(delta[1])
    plt.xticks(rotation = 45)
    plt.rcParams.update({'font.size': 12})
    sns.swarmplot(data = trj_hist, y = 'r_out',
                 x = 'lab', ax = delta[1], color='k')
    sns.boxplot(data = trj_hist, y = 'r_out',
                 x = 'lab', ax = delta[1], color='gray')
    delta[1].set_xlabel('Laboratory Location')
    delta[1].spines['top'].set_visible(False)
    delta[1].spines['right'].set_visible(False)
    delta[1].set_ylabel('Fraction of regions missed')
    
    return delta
    
def penetration_query():
    '''
    Currently ONE dependent, in the future it will accept DJ query
    INPUT
        None
    OUTPUT
        Return pandas dataframe with information from each penentration
    '''
    
    # Connect to ONE and fetch data from histology
    one = ONE(base_url='https://alyx.internationalbrainlab.org') 
    trajectories_hist = one.alyx.rest('trajectories', 'list', 
                             provenance='Histology Track')
   
    # Make Dataframe
    trj_hist = pd.DataFrame(trajectories_hist)
    ses_hist = pd.DataFrame(list(trj_hist['session'][:]))
    trj_hist = pd.concat([trj_hist,ses_hist],axis = 1)
    trj_hist = trj_hist.drop(columns= ['session'])
    
    # For sessions with histology obtain micro-manipulator data and add to df
    # Parameter probe_insertion is the same for planned,mcr and hist from same
    # penetration.
    trj_hist['mcr_depth'] = np.nan
    trj_hist['mcr_phi'] = np.nan
    trj_hist['mcr_roll'] = np.nan
    trj_hist['mcr_theta'] = np.nan
    trj_hist['mcr_x'] = np.nan
    trj_hist['mcr_y'] = np.nan
    trj_hist['mcr_z'] = np.nan
    trj_hist['pln_depth'] = np.nan
    trj_hist['pln_phi'] = np.nan
    trj_hist['pln_roll'] = np.nan
    trj_hist['pln_theta'] = np.nan
    trj_hist['pln_x'] = np.nan
    trj_hist['pln_y'] = np.nan
    trj_hist['pln_z'] = np.nan
    
    # ONE does not accept list of string for fetching, looping instead
    for i in range(len(trj_hist['probe_insertion'])):
        trj = pd.DataFrame(one.alyx.rest('trajectories', 
                'list', probe_insertion = str(trj_hist['probe_insertion'][i])))
        try:
            micro =  trj.loc[trj['provenance'] == 'Micro-manipulator']
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'mcr_depth'] = float(micro['depth'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'mcr_phi'] =  float(micro['phi'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'mcr_roll'] =  float(micro['roll'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'mcr_theta'] =  float(micro['theta'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'mcr_x'] =  float(micro['x'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'mcr_y'] =  float(micro['y'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'mcr_z'] = float(micro['z'])
        
        except:
            pass

        try:
            pln =  trj.loc[trj['provenance'] == 'Planned']
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_depth'] = float(pln['depth'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_phi'] = float(pln['phi'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_roll'] = float(pln['roll'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_theta'] = float(pln['theta'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_x'] = float(pln['x'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_y'] = float(pln['y'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_z'] = float(pln['z'])
        except:
            pass
    
    return trj_hist


def fig_delta_coordinates(trj_hist):
    '''
    Plots the difference between the micromanipulator coordinates and the allen 
    coordinates from histology
    INPUT:
        trj_hist: Dataframe with data from penetrations
    OUTPUT:
        Figure with error 
    '''
   
    figure, delta = plt.subplots(2,2, figsize =[20,20])
    plt.sca(delta[0,0])
    plt.xticks(rotation = 45)
    plt.rcParams.update({'font.size': 12})
    sns.swarmplot(data = trj_hist, y = trj_hist['depth'] - trj_hist['mcr_depth'],
                 x = 'lab', ax = delta[0,0], color='k')
    sns.boxplot(data = trj_hist, y = trj_hist['depth'] - trj_hist['mcr_depth'],
                 x = 'lab', ax = delta[0,0], color='gray')
    delta[0,0].set_xlabel('Laboratory Location')
    delta[0,0].spines['top'].set_visible(False)
    delta[0,0].spines['right'].set_visible(False)
    delta[0,0].set_ylabel('Depth error (um)')
    plt.sca(delta[0,1])
    plt.xticks(rotation = 45)
    plt.rcParams.update({'font.size': 12})
    sns.swarmplot(data = trj_hist, y = trj_hist['theta'] - trj_hist['mcr_theta'],
                 x = 'lab', color='k')
    sns.boxplot(data = trj_hist, y = trj_hist['theta'] - trj_hist['mcr_theta'],
                 x = 'lab', color='gray')
    delta[0,1].set_xlabel('Laboratory Location')
    delta[0,1].spines['top'].set_visible(False)
    delta[0,1].spines['right'].set_visible(False)
    delta[0,1].set_ylabel('Theta angle error (degrees)')
    plt.sca(delta[1,0])
    plt.xticks(rotation = 45)
    plt.rcParams.update({'font.size': 12})
    sns.swarmplot(data = trj_hist, y = trj_hist['x'] - trj_hist['mcr_x'],
                 x = 'lab', color='k')
    sns.boxplot(data = trj_hist, y = trj_hist['x'] - trj_hist['mcr_x'],
                 x = 'lab', color='gray')
    delta[1,0].set_xlabel('Laboratory Location')
    delta[1,0].spines['top'].set_visible(False)
    delta[1,0].spines['right'].set_visible(False)
    delta[1,0].set_ylabel('ML error (um)')
    plt.sca(delta[1,1])
    plt.xticks(rotation = 45)
    plt.rcParams.update({'font.size': 12})
    sns.swarmplot(data = trj_hist, y = trj_hist['y'] - trj_hist['mcr_y'],
                 x = 'lab', color='k')
    sns.boxplot(data = trj_hist, y = trj_hist['y'] - trj_hist['mcr_y'],
                 x = 'lab', color='gray')
    delta[1,1].set_xlabel('Laboratory Location')
    delta[1,1].spines['top'].set_visible(False)
    delta[1,1].spines['right'].set_visible(False)
    delta[1,1].set_ylabel('AP error (um)')

def patch_name(dataframe):
    '''
    Patches names from ONE or DJ, and turns them into legible "official" names
    INPUT:
        dataframe: dataframe with data containing varaible with labname
    OUTPUT:
        dataframe with labname fixedd
        
    '''
    dataframe['lab'] =  dataframe['lab'].map({'churchlandlab': 'Churchland Lab', 
                                'cortexlab': 'Carandini-Harris Lab', 
                                'mainenlab': 'Mainen Lab', 
                                'mrsicflogellab': 'Mrsic-Flogel Lab',
                                'hoferlab':  'Hofer Lab', 
                                'wittenlab': ' Witten Lab', 
                                'zadorlab':  'Zador Lab'})
    return dataframe

if _main_ == _name_:
     ## Get dataframe with data
     trj_hist = penetration_query()
     ## Fix the name
     trj_hist = patch_name(trj_hist)
     ## Figure with error in penetration depth
     fig_delta_coordinates(trj_hist)
     # Figure with cumulative error between two probes
     manipulator_vs_histology(trj_hist)
     planed_vs_histology(trj_hist)
     
     
    
     
     

     
     
     
     
     
    
    