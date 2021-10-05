#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jun  1 09:11:26 2020

Probe Geometry Plots

Functions to plot slices of histology data along defined probes, and add probe
trajectories or channel coords to these plots.


Following functions are defined:



@author: stevenwest
'''


def plot_probe_trajectory_atlas_coronal(x, y,
                                        project = 'ibl_neuropixel_brainwide_01',
                                        remove_primary_axis = False):
    """Plot coronal slice of Atlas along the planned probe trajectory.
    
    Subject_ID is 'Planned' by default, in which case the planned probe trajectory
    is used for planned coord [x,y] to generate a 2D slice through the Atlas.
    
    Subject_ID can be set to any VALID ID that contains a planned trajectory at
    coord [x,y].  This is useful if a different provenance of the trajectory
    is used for plotting the 2D slice through atlas (Micro-Manipulator, Histology
    track, Ephys aligned histology track)
    
    
    """
    from one.api import ONE
    import ibllib.atlas as atlas
    from ibllib.atlas import Insertion
    import atlaselectrophysiology.load_histology as hist
    import numpy as np
    
    import sys
    
    import matplotlib.pyplot as plt
    
    # connect to ONE
    one = ONE()
    
    # get list of all trajectories at planned [x,y], for project
    trajs = one.alyx.rest('trajectories', 'list', x=x, y=y,  project=project)
    
    # get insertion object from ANY (the first) trajectory
    ins = Insertion.from_dict(trajs[0])
    
    axis_labels = np.array(['ml (µm)', 'dv (µm)', 'ap (µm)'])
    fig1, ax1 = plt.subplots() # new figure and axes objects - CORONAL
    
    # get CCF brain atlas for generating the figures:
    brain_atlas = atlas.AllenAtlas(res_um=25)
    
    ax1 = brain_atlas.plot_tilted_slice(ins.xyz, axis=1, ax=ax1) # CORONAL
    
    if remove_primary_axis:
        ax1.get_yaxis().set_visible(False)
    
    return fig1 # return the coronal plot!
    



def plot_probe_trajectory_atlas_sagittal(trajectory):
    """Plot sagittal slice of Atlas along the probe trajectory.
    
    The slice through the Atlas can be made along any of the provenances of
    the probe at [x,y] for subject ID - Planned, Micro-manipulator, Histology
    track, Ephys aligned histology track.
    
    """
    from one.api import ONE
    import ibllib.atlas as atlas
    from ibllib.atlas import Insertion
    import atlaselectrophysiology.load_histology as hist
    import numpy as np
    
    import sys
    
    import matplotlib.pyplot as plt
    
    # connect to ONE
    one = ONE()
    



def plot_probe_trajectory_histology_coronal(
        x, y, subject_ID, 
        provenance = 'Planned', project = 'ibl_neuropixel_brainwide_01',
        remove_primary_axis = False ):
    """Plot slice of Histology data along the insertion at [x,y] for subject ID.
    
    The slice through the Histology data can be made along any of the 
    provenances of the probe at [x,y] for subject ID - Planned, 
    Micro-manipulator, Histology track, Ephys aligned histology track.
    
    """
    
    from one.api import ONE
    import ibllib.atlas as atlas
    from ibllib.atlas import Insertion
    import atlaselectrophysiology.load_histology as hist
    import numpy as np
    
    import sys
    
    import matplotlib.pyplot as plt
    
    # connect to ONE
    one = ONE()
    
    # get list of all trajectories at planned [x,y], for project
    trajs = one.alyx.rest('trajectories', 'list', x=x, y=y,  project=project)
    
    # get insertion object from ANY (the first) trajectory
    ins = Insertion.from_dict(trajs[0])
    
    axis_labels = np.array(['ml (µm)', 'dv (µm)', 'ap (µm)'])
    fig1, ax1 = plt.subplots() # new figure and axes objects - CORONAL
    
    
    



def plot_probe_atlas_rep_site(x, y, ):
    



def plot_probe_histology_rep_site(x,y,):
    
    

