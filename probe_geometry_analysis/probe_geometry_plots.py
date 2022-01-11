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
    
    A tilted coronal slice of Atlas is made along planned probe trajectory,
    at [x,y], from project.
    
    
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
    trajs = one.alyx.rest('trajectories', 'list', x=x, y=y,  project=project,
                          provenance='Planned')
    
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
    



def plot_probe_trajectory_atlas_sagittal(x, y,
                                        project = 'ibl_neuropixel_brainwide_01',
                                        remove_primary_axis = False):
    """Plot sagittal slice of Atlas along the planned probe trajectory.
    
    A tilted sagittal slice of Atlas is made along planned probe trajectory,
    at [x,y], from project.
    
    
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
    trajs = one.alyx.rest('trajectories', 'list', x=x, y=y,  project=project,
                          provenance='Planned')
    
    # get insertion object from ANY (the first) trajectory
    ins = Insertion.from_dict(trajs[0])
    
    axis_labels = np.array(['ml (µm)', 'dv (µm)', 'ap (µm)'])
    fig1, ax1 = plt.subplots() # new figure and axes objects - SAGITTAL
    
    # get CCF brain atlas for generating the figures:
    brain_atlas = atlas.AllenAtlas(res_um=25)
    
    ax1 = brain_atlas.plot_tilted_slice(ins.xyz, axis=0, ax=ax1) # SAGITTAL
    
    if remove_primary_axis:
        ax1.get_yaxis().set_visible(False)
    
    return fig1 # return the sagittal plot!




def get_insertion_from_trajectory(trajs, subject_ID, provenance):
    
    from one.api import ONE
    from ibllib.atlas import Insertion
    
    # connect to ONE
    one = ONE()
    
    # keeping subjs and labs for look-up later if needed..
    subjs = [sess['session']['subject'] for sess in trajs]
    #aidx = subjs.index(atlas_ID)
    sidx = subjs.index(subject_ID)
    
    # Fetch trajectory metadata for traj:
    traj = one.alyx.rest('trajectories', 'list', session=trajs[sidx]['session']['id'],
                 probe=trajs[sidx]['probe_name'], provenance=provenance)
    
    if traj == []:
        raise Exception("No trajectory found with provenance: " + provenance)
    
    # get insertion object from ANY (the first) trajectory
    ins = Insertion.from_dict(traj[0])
    
    return ins


def plot_probe_trajectory_histology(
        x, y, subject_ID, axc, axs, 
        provenance = 'Planned', 
        project = 'ibl_neuropixel_brainwide_01',
        gr_percentile_min=0.2, rd_percentile_min=1, rd_percentile_max=99.99, 
        font_size = 8, label_size = 8 ):
    """Plot slices of Histology data along the insertion at [x,y] for subject ID.
    
    Slices made in coronal and sagittal planes.
    
    The slices through the Histology data can be made along any of the 
    provenances of the probe at [x,y] for subject ID - Planned, 
    Micro-manipulator, Histology track, Ephys aligned histology track.
    
    axc : AxesSubplot, None
        MUST pass an AxesSubplot object for plotting to!  For coronal plot.
    
    axs : AxesSubplot, None
        MUST pass an AxesSubplot object for plotting to!  For sagittal plot.
    
    """
    
    from one.api import ONE
    import ibllib.atlas as atlas
    from ibllib.atlas import Insertion
    import atlaselectrophysiology.load_histology as hist
    import numpy as np
    from scipy import ndimage
    
    import sys
    
    import matplotlib.pyplot as plt
    
    # connect to ONE
    one = ONE()
    
    # get list of all trajectories at [x,y], for project
    trajs = one.alyx.rest('trajectories', 'list', x=x, y=y,  project=project)
    
    # keeping subjs and labs for look-up later if needed..
    subjs = [sess['session']['subject'] for sess in trajs]
    labs = [sess['session']['lab'] for sess in trajs]
    #aidx = subjs.index(atlas_ID)
    sidx = subjs.index(subject_ID)
    
    # Fetch trajectory metadata for traj:
    traj = one.alyx.rest('trajectories', 'list', session=trajs[sidx]['session']['id'],
                 probe=trajs[sidx]['probe_name'], provenance=provenance)
    
    if traj == []:
        raise Exception("No trajectory found with provenance: " + provenance)
    
    # get insertion object from ANY (the first) trajectory
    ins = Insertion.from_dict(traj[0])
    
    axis_labels = np.array(['ml (µm)', 'dv (µm)', 'ap (µm)'])
    
    #fig1, ax1 = plt.subplots() # new figure and axes objects - CORONAL
    #fig2, ax2 = plt.subplots() # new figure and axes objects - SAGITTAL
    
    # set axes to local variables
    ax1 = axc
    ax2 = axs
    
    lab = labs[ sidx ] # this returns index in labs where subject_ID is in subjs
    hist_paths = hist.download_histology_data(subject_ID, lab)
    
    # create the brain atlases from the data
    ba_gr = atlas.AllenAtlas(hist_path=hist_paths[0]) # green histology channel autofl.
    ba_rd = atlas.AllenAtlas(hist_path=hist_paths[1]) # red histology channel cm-dii
    
    # CORONAL
    
    # implementing tilted slice here to modify its cmap
     # get tilted slice of the green and red channel brain atlases
      # using the .image data as this contains the signal
    gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 1, volume = ba_gr.image)
    rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 1, volume = ba_rd.image)
    
    gr_tslice_roi = gr_tslice[120:240, 150:300] # isolate large slice over thalamus for max pixel value
    rd_tslice_roi = rd_tslice[120:240, 150:300]
    
    width = width * 1e6
    height = height * 1e6
    depth = depth * 1e6
    
    cmap = plt.get_cmap('bone')
    
    # get the transfer function from y-axis to squeezed axis for second axe
    ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
    height * ab[0] + ab[1]
    
     # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
      # Using gr_tslice min and gr_tslice_roi max to scale autofl.
      # using rd_tslice min and percentile (99.99 default) to scale CM-DiI
    gr_in = np.interp(gr_tslice, (np.percentile(gr_tslice, gr_percentile_min), gr_tslice_roi.max()), (0, 255))
    rd_in = np.interp(rd_tslice, (np.percentile(rd_tslice, rd_percentile_min), np.percentile(rd_tslice, rd_percentile_max)), (0, 255))
    
     # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
      # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
      # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
    Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                   gr_in.astype(dtype=np.uint8), 
                   np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
     # transpose the columns to the FIRST one is LAST 
     # i.e the NEW DIMENSION [3] is the LAST DIMENSION
    Zt = np.transpose(Z, axes=[1,2,0])
    
     # can now add the RGB array to imshow()
    ax1.imshow(Zt, interpolation='none', aspect='auto', extent=np.r_[width, height], cmap=cmap, vmin=np.min(gr_in), vmax=np.max(gr_in) )
    
    sec_ax = ax1.secondary_yaxis('right', functions=(
                        lambda x: x * ab[0] + ab[1],
                        lambda y: (y - ab[1]) / ab[0]))
    
    ax1.set_xlabel(axis_labels[0], fontsize=font_size)
    ax1.set_ylabel(axis_labels[1], fontsize=font_size)
    sec_ax.set_ylabel(axis_labels[2], fontsize=font_size)
    
    ax1.tick_params(axis='x', labelrotation = 90)
    
    ax1.tick_params(axis='x', labelsize = label_size)
    ax1.tick_params(axis='y', labelsize = label_size)
    sec_ax.tick_params(axis='y', labelsize = label_size)
    
    # SAGITTAL
    
    # implementing tilted slice here to modify its cmap
     # get tilted slice of the green and red channel brain atlases
      # using the .image data as this contains the signal
    gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 0, volume = ba_gr.image)
    rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 0, volume = ba_rd.image)
    
    width = width * 1e6
    height = height * 1e6
    depth = depth * 1e6
    
    cmap = plt.get_cmap('bone')
    
    # get the transfer function from y-axis to squeezed axis for second axe
    ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
    height * ab[0] + ab[1]
    
     # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
      # Using gr_tslice min and max to scale the image
       # weirdly rd_in has very large min and max (problem with the original data acquisition?) so best to scale whole RGB with gr_in/1.5!
    gr_in = np.interp(gr_tslice, (gr_tslice.min(), gr_tslice.max()), (0, 255))
    rd_in = np.interp(rd_tslice, (gr_tslice.min(), gr_tslice.max()/1.5), (0, 255))
    
     # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
      # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
      # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
    Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                   gr_in.astype(dtype=np.uint8), 
                   np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
     # transpose the columns to the FIRST one is LAST 
     # i.e the NEW DIMENSION [3] is the LAST DIMENSION
    Zt = np.transpose(Z, axes=[1,2,0])
    
     # can now add the RGB array to ax2 via imshow()
    ax2.imshow(Zt, interpolation='none', aspect='auto', extent=np.r_[width, height], cmap=cmap, vmin=np.min(gr_in), vmax=np.max(gr_in) )
    
    #start = ins.xyz[:, 1] * 1e6
    #end = ins.xyz[:, 2] * 1e6
    #xCoords = np.array([start[0], end[0]])
    
    sec_ax = ax2.secondary_yaxis('right', functions=(
                        lambda x: x * ab[0] + ab[1],
                        lambda y: (y - ab[1]) / ab[0]))
    
    ax2.set_xlabel(axis_labels[2], fontsize=font_size)
    ax2.set_ylabel(axis_labels[1], fontsize=font_size)
    sec_ax.set_ylabel(axis_labels[0], fontsize=font_size)
    
    ax2.tick_params(axis='x', labelrotation = 90)
    
    ax2.tick_params(axis='x', labelsize = label_size)
    ax2.tick_params(axis='y', labelsize = label_size)
    sec_ax.tick_params(axis='y', labelsize = label_size)
    
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    # add a line of the Insertion object onto ax1 (cax - coronal)
     # plotting PLANNED insertion 
    #ax1.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
    #ax2.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
    
    return  {'coronal-slice': ax1, 'sagittal-slice': ax2, 'x': x, 'y': y, 
             'provenance': provenance, 'subject_id': subject_ID }
    


def plot_channels(fig_dict, colour='y'):
    """Plot subject channels
    
    Generates coronal and sagittal plots of the planned repeated site trajectory,
    plus points for each channel projected onto each figure for the specified
    subject_ID.
    
    Parameters
    ----------
    fig_dict : dictionary
        Dictionary containing the pyplot Figure objects for coronal-slice and
        cagittal-slice, and the insertion coordinates x & y.  Get initial plot
        from plot_probe_trajectory..() functions
        
    colour : str, optional
        String code for colour of the pyplot line (y, g, r, b, k, w, etc). 
        The default is 'y'.

    Returns
    -------
    fig_dict : dictionary
        Dictionary containing the pyplot Figure objects for coronal (cax) and
        cagittal (sax).

    """
    from probe_geometry_analysis import probe_geometry_data as ch_data
    import matplotlib.pyplot as plt
    
    # get repeated site ch disp data    
    data_frame = ch_data.load_channels_data( str(fig_dict['x']), str(fig_dict['y']) )
    subject_ID = fig_dict['subject_id']
    
    # subset the data_frame to subject
    subj_frame = data_frame[data_frame['subject'] == subject_ID]
    
    # retrieve the location in XYZ
    locX = subj_frame['chan_loc_x'].values
    locY = subj_frame['chan_loc_y'].values
    locZ = subj_frame['chan_loc_z'].values
    
    # get the axes:
    cax = fig_dict['coronal-slice']
    sax = fig_dict['sagittal-slice']
    
    # create generic plt fig
    fig, ax = plt.subplots()
    
    # plot channels as circles at hald the dpi
     # this gives channel coords that are just about separate in the figure!
    cax.plot(locX * 1e6, locZ * 1e6,  marker='o',
             ms=(72./fig.dpi)/2, mew=0, 
        color=colour, linestyle="", lw=0)
    
    sax.plot(locY * 1e6, locZ * 1e6, marker='o',
             ms=(72./fig.dpi)/2, mew=0, 
        color=colour, linestyle="", lw=0)
    
    return fig_dict


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
    from scipy import ndimage
    
    import sys
    
    import matplotlib.pyplot as plt
    
    # connect to ONE
    one = ONE()
    
    # get list of all trajectories at [x,y], for project
    trajs = one.alyx.rest('trajectories', 'list', x=x, y=y,  project=project)
    
    # keeping subjs and labs for look-up later if needed..
    subjs = [sess['session']['subject'] for sess in trajs]
    labs = [sess['session']['lab'] for sess in trajs]
    #aidx = subjs.index(atlas_ID)
    sidx = subjs.index(subject_ID)
    
    # Fetch trajectory metadata for traj:
    traj = one.alyx.rest('trajectories', 'list', session=trajs[sidx]['session']['id'],
                 probe=trajs[sidx]['probe_name'], provenance=provenance)
    
    if traj == []:
        raise Exception("No trajectory found with provenance: " + provenance)
    
    # get insertion object from ANY (the first) trajectory
    ins = Insertion.from_dict(traj[0])
    
    axis_labels = np.array(['ml (µm)', 'dv (µm)', 'ap (µm)'])
    fig1, ax1 = plt.subplots() # new figure and axes objects - CORONAL
    
    lab = labs[ sidx ] # this returns index in labs where subject_ID is in subjs
    hist_paths = hist.download_histology_data(subject_ID, lab)
    
    # create the brain atlases from the data
    ba_gr = atlas.AllenAtlas(hist_path=hist_paths[0]) # green histology channel autofl.
    ba_rd = atlas.AllenAtlas(hist_path=hist_paths[1]) # red histology channel cm-dii
    
    # implementing tilted slice here to modify its cmap
     # get tilted slice of the green and red channel brain atlases
      # using the .image data as this contains the signal
    gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 1, volume = ba_gr.image)
    rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 1, volume = ba_rd.image)
    
    # gaussian filtered image of the whole 3D GR stack
    gr_g4 = ndimage.gaussian_filter(ba_gr.image, 4)
    
    width = width * 1e6
    height = height * 1e6
    depth = depth * 1e6
    
    cmap = plt.get_cmap('bone')
    
    # get the transfer function from y-axis to squeezed axis for second axe
    ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
    height * ab[0] + ab[1]
    
     # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
      # Using MEDIAN FILTERED (radius 3) gr_tslice min and max to scale the image
       # weirdly rd_in has very large min and max (problem with the original data acquisition?) so best to scale whole RGB with gr_in/1.5!
    #gr_ts_m3 = ndimage.median_filter(gr_tslice, 3)
    #gr_ts_m6 = ndimage.median_filter(gr_tslice, 6)
    #gr_ts_m8 = ndimage.median_filter(gr_tslice, 8)
    #rd_ts_m3 = ndimage.median_filter(rd_tslice, 3)
    #gr_in = np.interp(gr_tslice, (gr_ts_m3.min(), gr_ts_m3.max()), (0, 255))
    #gr_in = np.interp(gr_tslice, (gr_m4.min(), gr_m4.max()), (0, 255))
    #gr_in = np.interp(gr_tslice, (gr_ts_m8.min(), gr_ts_m8.max()), (0, 255))
    gr_in = np.interp(gr_tslice, (gr_g4.min(), gr_g4.max()), (0, 255))
    rd_in = np.interp(rd_tslice, (rd_tslice.min(), rd_tslice.max()/3), (0, 255))
    
    # original normalisation
    gr_in = np.interp(gr_tslice, (gr_tslice.min(), gr_tslice.max()), (0, 255))
    rd_in = np.interp(rd_tslice, (gr_tslice.min(), gr_tslice.max()/1.5), (0, 255))
    
     # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
      # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
      # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
    Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                   gr_in.astype(dtype=np.uint8), 
                   np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
     # transpose the columns to the FIRST one is LAST 
     # i.e the NEW DIMENSION [3] is the LAST DIMENSION
    Zt = np.transpose(Z, axes=[1,2,0])
    
     # can now add the RGB array to imshow()
    ax1.imshow(Zt, interpolation='none', aspect='auto', extent=np.r_[width, height], cmap=cmap, vmin=np.min(gr_in), vmax=np.max(gr_in) )
    
    sec_ax = ax1.secondary_yaxis('right', functions=(
                        lambda x: x * ab[0] + ab[1],
                        lambda y: (y - ab[1]) / ab[0]))
    ax1.set_xlabel(axis_labels[0], fontsize=8)
    ax1.set_ylabel(axis_labels[1], fontsize=8)
    sec_ax.set_ylabel(axis_labels[2], fontsize=8)
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    return fig1
    


def plot_probe_trajectory_histology_sagittal(
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
    from scipy import ndimage
    
    import sys
    
    import matplotlib.pyplot as plt
    
    # connect to ONE
    one = ONE()
    
    # get list of all trajectories at [x,y], for project
    trajs = one.alyx.rest('trajectories', 'list', x=x, y=y,  project=project)
    
    # keeping subjs and labs for look-up later if needed..
    subjs = [sess['session']['subject'] for sess in trajs]
    labs = [sess['session']['lab'] for sess in trajs]
    #aidx = subjs.index(atlas_ID)
    sidx = subjs.index(subject_ID)
    
    # Fetch trajectory metadata for traj:
    traj = one.alyx.rest('trajectories', 'list', session=trajs[sidx]['session']['id'],
                 probe=trajs[sidx]['probe_name'], provenance=provenance)
    
    if traj == []:
        raise Exception("No trajectory found with provenance: " + provenance)
    
    # get insertion object from ANY (the first) trajectory
    ins = Insertion.from_dict(traj[0])
    
    axis_labels = np.array(['ml (µm)', 'dv (µm)', 'ap (µm)'])
    fig1, ax1 = plt.subplots() # new figure and axes objects - SAGITTAL
    
    lab = labs[ sidx ] # this returns index in labs where subject_ID is in subjs
    hist_paths = hist.download_histology_data(subject_ID, lab)
    
    # create the brain atlases from the data
    ba_gr = atlas.AllenAtlas(hist_path=hist_paths[0]) # green histology channel autofl.
    ba_rd = atlas.AllenAtlas(hist_path=hist_paths[1]) # red histology channel cm-dii
    
    # implementing tilted slice here to modify its cmap
     # get tilted slice of the green and red channel brain atlases
      # using the .image data as this contains the signal
    gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 0, volume = ba_gr.image)
    rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 0, volume = ba_rd.image)
    
    # gaussian filtered image of the whole 3D GR stack
    gr_g4 = ndimage.gaussian_filter(ba_gr.image, 4)
    
    ## SET CONTRAST based on min() and max() of hippocampus & thalamus pixels ONLY
      # use the gaussian blurred image for pixel values, use ba_gr.label
    
    width = width * 1e6
    height = height * 1e6
    depth = depth * 1e6
    
    cmap = plt.get_cmap('bone')
    
    # get the transfer function from y-axis to squeezed axis for second axe
    ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
    height * ab[0] + ab[1]
    
     # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
      # Using MEDIAN FILTERED (radius 3) gr_tslice min and max to scale the image
       # weirdly rd_in has very large min and max (problem with the original data acquisition?) so best to scale whole RGB with gr_in/1.5!
    #gr_ts_m3 = ndimage.median_filter(gr_tslice, 3)
    #gr_ts_m6 = ndimage.median_filter(gr_tslice, 6)
    #gr_ts_m8 = ndimage.median_filter(gr_tslice, 8)
    #rd_ts_m3 = ndimage.median_filter(rd_tslice, 3)
    #gr_in = np.interp(gr_tslice, (gr_ts_m3.min(), gr_ts_m3.max()), (0, 255))
    #gr_in = np.interp(gr_tslice, (gr_m4.min(), gr_m4.max()), (0, 255))
    #gr_in = np.interp(gr_tslice, (gr_ts_m8.min(), gr_ts_m8.max()), (0, 255))
    gr_in = np.interp(gr_tslice, (gr_g4.min(), gr_g4.max()), (0, 255))
    rd_in = np.interp(rd_tslice, (rd_tslice.min(), rd_tslice.max()/3), (0, 255))
    
    # original normalisation
    gr_in = np.interp(gr_tslice, (gr_tslice.min(), gr_tslice.max()), (0, 255))
    rd_in = np.interp(rd_tslice, (gr_tslice.min(), gr_tslice.max()/1.5), (0, 255))
    
     # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
      # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
      # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
    Z = np.stack([ rd_in.astype(dtype=np.uint8), 
                   gr_in.astype(dtype=np.uint8), 
                   np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
     # transpose the columns to the FIRST one is LAST 
     # i.e the NEW DIMENSION [3] is the LAST DIMENSION
    Zt = np.transpose(Z, axes=[1,2,0])
    
     # can now add the RGB array to imshow()
    ax1.imshow(Zt, interpolation='none', aspect='auto', extent=np.r_[width, height], cmap=cmap, vmin=np.min(gr_in), vmax=np.max(gr_in) )
    
    sec_ax = ax1.secondary_yaxis('right', functions=(
                        lambda x: x * ab[0] + ab[1],
                        lambda y: (y - ab[1]) / ab[0]))
    ax1.set_xlabel(axis_labels[0], fontsize=8)
    ax1.set_ylabel(axis_labels[1], fontsize=8)
    sec_ax.set_ylabel(axis_labels[2], fontsize=8)
    
    #xmn = np.min(xCoords) - 500
    #xmz = np.max(xCoords) + 500
    xmn = np.min(ins.xyz[:, 0]) * 1e6 - 1000
    xmz = np.max(ins.xyz[:, 0]) *1e6 + 1000
    
    ax1.set_xlim(xmn, xmz)
     # ensure the resized xlim is not stretched!
    ax1.axes.set_aspect('equal')
    ax1.tick_params(axis='x', labelrotation = 90)
    
    ax1.tick_params(axis='x', labelsize = 8)
    ax1.tick_params(axis='y', labelsize = 8)
    sec_ax.tick_params(axis='y', labelsize = 8)
    
    if remove_primary_axis:
        #ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    return fig1
    


#def plot_probe_atlas_rep_site(x, y, ):



#def plot_probe_histology_rep_site(x,y,):

