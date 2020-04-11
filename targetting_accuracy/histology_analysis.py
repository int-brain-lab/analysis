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


# Function definitions

def planed_vs_manipulator(trj_planned, trj_manipulator):
    '''
    Measures the error between the planned probe insertion and the one recorded
    in the manipulator on recording day.
    INPUT:
        trj_planned: Planned trajectory
        trj_manipulator: Micromanipulator trajectory
    OUTPUT:
        cecle: Comulative ecludian error

    '''

def manipulator_vs_histology(trj_theory, trj_hist):
    '''
    Measures the error between the manipulator or planned probe insertion 
    and the one from histology
    INPUT:
        trj_theory: Planned trajectory
        trj_hist: Histology trajectory
    OUTPUT:
        cecle: Comulative ecludian error
        
    '''
    
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
            ttrj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'mcr_z'] = float(micro['z'])
        
        except:
            pass

        try:
            pln =  trj.loc[trj['provenance'] == 'Planned']
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_depth'] = float(pln['depth'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_phi'] =  float(pln['phi'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_roll'] =  float(pln['roll'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_theta'] =  float(pln['theta'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_x'] =  float(pln['x'])
            trj_hist.loc[trj_hist['probe_insertion'] == \
                 trj_hist['probe_insertion'][i] ,'pln_y'] =  float(pln['y'])
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

def cumulative_error(trj_hist):
    '''
    Calculates the cumulative error between the tracked histology and the
    recorded location in the micromanipulator
    INPUT:
        trj_hist : pandas dataframe with penetration locations
    OUTPUT:
        figure with cumulative error per lab
    '''


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


if __name__ == __main__:
     ## Get dataframe with data
     trj_hist = penetration_query()
     ## Fix the name
     trj_hist = patch_name(trj_hist)
     ## Figure with error in penetration depth
     fig_delta_coordinates(trj_hist)
     # Figure with cumulative error between two probes
     cumulative_error(trj_hist)
     
     
     
    
     
     

     
     
     
     
     
    
    