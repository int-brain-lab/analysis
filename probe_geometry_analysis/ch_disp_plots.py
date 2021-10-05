#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jun  1 09:11:26 2020

Channels Displacement Plot Generation

Generate plots of channel displacement data from planned traj at repeated site.

This module defines functions for plotting the channels and planned trajectory
on 2D slices on AllenAtlas CCF.

Gather channels displacement data from planned traj at repeated site.

Following data is gathered:

* subject, eid, probe - the subject, eid and probe IDs

* chan_loc - xyz coord of channels

* planned_orth_proj - xyz coord of orthogonal line from chan_loc to planned proj

* dist - the 3D distance between chan_loc xyz and planned_orth_proj xyz

Data is saved to CSV next to this module.

NB: Channels are derived from the serializer: histology processed, histology,
micro manipulator, planned - the first is always returned.

@author: stevenwest
'''



def plot_atlas_traj(x, y, 
                    atlas_ID = 'CCF',
                    provenance='Planned', 
                    subject_ID = 'Planned', 
                    remove_primary_axis = False,
                    project='ibl_neuropixel_brainwide_01', 
                    altas_borders = False, 
                    colour='y', linewidth=1,
                    axc = None, axs = None ):
    """Plot atlas trajectory
    
    Generates coronal and sagittal plots of the atlas along trajectory at x,y.

    Parameters
    ----------
    
    x : int
        x insertion coord in µm.  Eg. repeated site is -2243
    
    y : int
        y insertion coord in µm. Eg. repeated site is -2000.
    
    atlas_ID : str, optional
        Atlas data to plot channels on: 'CCF' - Allen CCF, or sample data. If 
        using sample data as atlas, must set this string to the subject ID
        to collect the data from flatiron. The default is 'CCF'.
    
    provenance : str, optional
        Trajectory provenance to use when generating the tilted slice. Choose 
        from: Planned, Micro-manipulator, Histology, E-phys aligned. The 
        default is 'Planned'.
    
    subject_ID : str, optional
        The subject which to retrieve the trajectory from. This trajectory 
        defines the tilted slice through the atlas. The default is 
        'Planned', which means the planned trajectory is used (ignoring the
        provenance option below!). Can be set to any valid subject_ID with
        a trajectory at x,y.
    
    remove_primary_axis : bool, optional
        Boolean whether to remove the primary y-axis labels from the plot.  Useful
        to remove if concatenating plots together.
    
    project : str, optional
        Project from which data should be retrieved. The default is 
        'ibl_neuropixel_brainwide_01'.
    
    atlas_borders : bool, optional
        Boolean whether to add the atlas borders to the atlas image. False by 
        default.
    
    axc : AxesSubplot, None
        MUST pass an AxesSubplot object for plotting to!  For coronal plot.
    
    axs : AxesSubplot, None
        MUST pass an AxesSubplot object for plotting to!  For sagittal plot.
    
    Returns
    -------
    
    fig_dict : dictionary
        Dictionary containing the pyplot Figure objects for coronal (cax) and
        cagittal (sax), and the insertion coordinates (x) and (y).
    
    """
    
    # TESTING:
    #x = -2243
    #y = -2000
    #atlas_ID = 'CSHL052'
    #provenance='Histology track'
    #remove_primary_axis = False
    #subject_ID = 'NYUU-12'
    #project='ibl_neuropixel_brainwide_01'
    #altas_borders = False
    #colour='y'
    #linewidth=1
    
    from one.api import ONE
    import ibllib.atlas as atlas
    from ibllib.atlas import Insertion
    import atlaselectrophysiology.load_histology as hist
    import numpy as np
    
    import sys
    
    import matplotlib.pyplot as plt
    
    # connect to ONE
    one = ONE()
    
    # get the PLANNED TRAJECTORY for x,y insertion
    trajs = one.alyx.rest('trajectories', 'list', x=x, y=y,  project=project)
    
    # generate pyplots of the brain:
    
    #fig1, ax1 = plt.subplots() # new figure and axes objects - CORONAL
    #fig2, ax2 = plt.subplots() # new figure and axes objects - SAGITTAL
    
    ax1 = axc
    ax2 = axs
    
    axis_labels = np.array(['ml (µm)', 'dv (µm)', 'ap (µm)'])
    
    if 'CCF' in atlas_ID: # want to use the CCF data for the plot
        
        if subject_ID == 'Planned':
            sidx=0
            provenance='Planned' # get the Planned trajectory from first subject!
        else:
            subjs = [sess['session']['subject'] for sess in trajs]
            labs = [sess['session']['lab'] for sess in trajs]
            sidx = subjs.index(subject_ID)
        
        # Fetch planned trajectory metadata for ANY of the traj in trajs (just use first index!):
        planned = one.alyx.rest('trajectories', 'list', session=trajs[sidx]['session']['id'],
                     probe=trajs[0]['probe_name'], provenance=provenance)
        
        # create insertion object from planned trajectory:
        ins = Insertion.from_dict(planned[0])
        
        # create a trajectory object from this insertion:
        traj = ins.trajectory
        
        # get CCF brain atlas for generating the figures:
        brain_atlas = atlas.AllenAtlas(res_um=25)
         # this is an instance of ibllib.atlas.atlas.AllenAtlas, sub-class of 
         # ibllib.atlas.atlas.BrainAtlas
         # contains:
         #        self.image: image volume (ap, ml, dv)
         #        self.label: label volume (ap, ml, dv)
         #        self.bc: atlas.BrainCoordinate object
         #        self.regions: atlas.BrainRegions object
         #        self.top: 2d np array (ap, ml) containing the z-coordinate (m) of the surface of the brain
         #        self.dims2xyz and self.zyz2dims: map image axis order to xyz coordinates order
        
        # use method in BrainAtlas to plot a tilted slice onto ax1:
        ax1 = brain_atlas.plot_tilted_slice(ins.xyz, axis=1, ax=ax1) # CORONAL
        ax2 = brain_atlas.plot_tilted_slice(ins.xyz, axis=0, ax=ax2) # SAGITTAL
        
        if remove_primary_axis:
            #ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            #ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
        
        
    else: # hopefully atlas_ID matches an ID in subjs! in which case, use THIS SUBJECTS FLUOESCENCE DATA!
        
        subject_ID = atlas_ID # ensure subject and atlas are the SAME
        # keeping subjs and labs for look-up later if needed..
        subjs = [sess['session']['subject'] for sess in trajs]
        labs = [sess['session']['lab'] for sess in trajs]
        aidx = subjs.index(atlas_ID)
        sidx = subjs.index(subject_ID)
        
        # Fetch trajectory metadata for traj:
        traj = one.alyx.rest('trajectories', 'list', session=trajs[sidx]['session']['id'],
                     probe=trajs[sidx]['probe_name'], provenance=provenance)
        
        if traj == []:
            raise Exception("No trajectory found with provenance: " + provenance)
        
        # create insertion object from planned trajectory:
        ins = Insertion.from_dict(traj[0])
        
        # download the data - using Mayo's method!
        lab = labs[ aidx ] # this returns index in labs where atlas_ID is in subjs
        hist_paths = hist.download_histology_data(atlas_ID, lab)
        
        # create the brain atlases from the data
        ba_gr = atlas.AllenAtlas(hist_path=hist_paths[0]) # green histology channel autofl.
        ba_rd = atlas.AllenAtlas(hist_path=hist_paths[1]) # red histology channel cm-dii
        
        # implementing tilted slice here to modify its cmap
         # get tilted slice of the green and red channel brain atlases
          # using the .image data as this contains the signal
        gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 1, volume = ba_gr.image)
        rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, 1, volume = ba_rd.image)
        
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
        
         # can now add the RGB array to imshow()
        ax2.imshow(Zt, interpolation='none', aspect='auto', extent=np.r_[width, height], cmap=cmap, vmin=np.min(gr_in), vmax=np.max(gr_in) )
        
        #start = ins.xyz[:, 1] * 1e6
        #end = ins.xyz[:, 2] * 1e6
        #xCoords = np.array([start[0], end[0]])
        
        sec_ax = ax2.secondary_yaxis('right', functions=(
                            lambda x: x * ab[0] + ab[1],
                            lambda y: (y - ab[1]) / ab[0]))
        ax2.set_xlabel(axis_labels[2], fontsize=8)
        ax2.set_ylabel(axis_labels[1], fontsize=8)
        sec_ax.set_ylabel(axis_labels[0], fontsize=8)
        
        xmn = np.min(ins.xyz[:, 1]) * 1e6 - 1000
        xmz = np.max(ins.xyz[:, 1]) *1e6 + 1000
        
        ax2.set_xlim(xmn, xmz)
         # ensure the resized xlim is not stretched!
        ax2.axes.set_aspect('equal')
        ax2.tick_params(axis='x', labelrotation = 90)
        
        ax2.tick_params(axis='x', labelsize = 8)
        ax2.tick_params(axis='y', labelsize = 8)
        sec_ax.tick_params(axis='y', labelsize = 8)
        
        if remove_primary_axis:
            #ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
        
    
    #plt.tight_layout() # tighten layout around xlabel & ylabel
    
    # add a line of the Insertion object onto ax1 (cax - coronal)
     # plotting PLANNED insertion 
    #ax1.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
    #ax2.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
    
    return  {'cax': ax1, 'sax': ax2, 'x': x, 'y': y, 
             'atlas_ID': atlas_ID, 'provenance': provenance, 
             'subject_id': subject_ID }



def plot_planned_traj_rep_site(colour='y', linewidth=1):
    """Plot REPEATED SITE planned trajectory
    
    Generates coronal and sagittal plots of the planned repeated site trajectory
    using the colour and linewidth params.

    Parameters
    ----------
    
    colour : str, optional
        String code for colour of the pyplot line (y, g, r, b, k, w, etc). 
        The default is 'y'.
    
    linewidth : int, optional
        Width of line in pixels. The default is 1.
    
    Returns
    -------
    
    fig_dict : dictionary
        Dictionary containing the pyplot Figure objects for coronal (cax) and
        cagittal (sax), and the insertion coordinates x & y.
    
    """
    return plot_atlas_traj(-2243, -2000, colour=colour, linewidth=linewidth)



def plot_histology_traj(subject_ID, x, y, 
                      provenance='Histology track', 
                      project='ibl_neuropixel_brainwide_01', 
                      atlas_type = 'sample-autofl', 
                      altas_borders = False, 
                      colour='y', linewidth=1):
    
    # TESTING:
    #subject_ID = 'NYU-12'
    #x = -2243
    #y = -2000
    #provenance='Histology track'
    #project='ibl_neuropixel_brainwide_01'
    #atlas_type = 'sample-autofl'
    #altas_borders = False
    #colour='y'
    #linewidth=1
    
    from one.api import ONE
    import ibllib.atlas as atlas
    from ibllib.atlas import Insertion
    import atlaselectrophysiology.load_histology as hist
    import numpy as np
    
    import matplotlib.pyplot as plt
    
    # connect to ONE
    one = ONE()
    
    # FIRST get the PLANNED TRAJECTORY for repeated site: x=2243, y=2000
     # first pass - to get a session id and probe name! - can retrieve from ANY trajectory!
    trajs = one.alyx.rest('trajectories', 'list',
                         x=x, y=y,  project=project)
    
    # get the lab string
    subjs = [sess['session']['subject'] for sess in trajs]
    labs = [sess['session']['lab'] for sess in trajs]
    
    lab = labs[ subjs.index(subject_ID)] # this returns index in labs where subject_ID is in subjs
    
    # Fetch Repeated Site planned trajectory metadata:
    planned = one.alyx.rest('trajectories', 'list', session=trajs[0]['session']['id'],
                 probe=trajs[0]['probe_name'], provenance='planned')
    
    # create insertion object of Repeated Site from planned trajectory:
    ins = Insertion.from_dict(planned[0])
    
    # create a trajectory object from this insertion:
    traj = ins.trajectory
    
    # generate pyplots of the brain:
    
    fig1, cax = plt.subplots() # new figure and axes objects - CORONAL
    fig2, sax = plt.subplots() # new figure and axes objects - SAGITTAL
    
    if 'sample-autofl' in atlas_type:
        
        # get just the autofl data as an atlas
        #TODO must modify this method!
        hist_paths = hist.download_histology_data(subject_ID, lab)
        
        ba_gr = atlas.AllenAtlas(hist_path=hist_paths[0]) # green histology channel autofl.
        #ba_rd = atlas.AllenAtlas(hist_path=hist_paths[1]) # red histology channel cm-dii
        
        # implementing tilted slice here to modify its cmap
         # get tilted slice of the green and red channel brain atlases
          # using the .image data as this contains the signal
        gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, 1, volume = ba_gr.image)
        
        width = width * 1e6
        height = height * 1e6
        depth = depth * 1e6
        
        cmap = plt.get_cmap('bone')
        
        # get the transfer function from y-axis to squeezed axis for second axe
        ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)
        height * ab[0] + ab[1]
        
         # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
          # Using gr_tslice min and max to scale the image
           # weirdly rd_in has very large min and max (problem with the original data acquisition?) so best to scale whole RGB with gr_in!
        gr_in = np.interp(gr_tslice, (gr_tslice.min(), gr_tslice.max()), (0, 255))
        
         # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
          # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
          # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
        Z = np.stack([ np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8), 
                      gr_in.astype(dtype=np.uint8), 
                      np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8) ])
         # transpose the columns to the FIRST one is LAST 
         # i.e the NEW DIMENSION [3] is the LAST DIMENSION
        Zt = np.transpose(Z, axes=[1,2,0])
        
         # can now add the RGB array to imshow()
        cax.imshow(Zt, interpolation='none', aspect='auto', extent=np.r_[width, height], cmap=cmap, vmin=np.min(gr_in), vmax=np.max(gr_in) )

        
    elif 'sample-cci' in atlas_type:
        
        print('sample-cci')
        
        
    elif 'sample' in atlas_type:
        
        print('sample')
        
    else:
        # invalid atlas choice - return error:
        print("INVALID ATLAS CHOICE - must be 'CCF' 'sample-autofl', 'sample-dii', 'sample'")
        #return None
    
    
    # add a line of the Insertion object onto ax1 (cax - coronal)
     # plotting PLANNED insertion 
    cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
    sax.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, colour, linewidth=linewidth)
    
    return  {'cax': fig1, 'sax': fig2, 'x': x, 'y': y}




def plot_subj_channels(fig_dict, colour='y'):
    """Plot subject channels
    
    Generates coronal and sagittal plots of the planned repeated site trajectory,
    plus points for each channel projected onto each figure for the specified
    subject_ID.
    
    Parameters
    ----------
    fig_dict : dictionary
        Dictionary containing the pyplot Figure objects for coronal (cax) and
        cagittal (sax), and the insertion coordinates x & y.  Get initial plot
        from plot_atlas_traj()
        
    colour : str, optional
        String code for colour of the pyplot line (y, g, r, b, k, w, etc). 
        The default is 'y'.

    Returns
    -------
    fig_dict : dictionary
        Dictionary containing the pyplot Figure objects for coronal (cax) and
        cagittal (sax).

    """
    import matplotlib.pyplot as plt
    
    # get repeated site ch disp data    
    data_frame = load_ch_disp_data( str(fig_dict['x']) + "_" + str(fig_dict['y']) )
    subject_ID = fig_dict['subject_id']
    
    # subset the data_frame to subject
    subj_frame = data_frame[data_frame['subject'] == subject_ID]
    
    # retrieve the location in XYZ
    locX = subj_frame['chan_loc_x'].values
    locY = subj_frame['chan_loc_y'].values
    locZ = subj_frame['chan_loc_z'].values
    
    # get the axes:
    cax = fig_dict['cax']
    sax = fig_dict['sax']
    
    # create generic plt fig
    fig, ax = plt.subplots()
    
    # plot channels as circles at hald the dpi
     # this gives channel coords that are just about separate in the figure!
    cax.plot(locX * 1e6, locZ * 1e6,  marker='o',ms=(72./fig.dpi)/2, mew=0, 
        color=colour, linestyle="", lw=0)
    
    sax.plot(locY * 1e6, locZ * 1e6, marker='o',ms=(72./fig.dpi)/2, mew=0, 
        color=colour, linestyle="", lw=0)
    
    return fig_dict


def plot_subj_channels_rep_site(subject_ID,
                      provenance='E-phys aligned', 
                      project='ibl_neuropixel_brainwide_01', 
                      colour='r'):
    
    return plot_subj_channels(-2243, -2000, subject_ID, provenance=provenance,
                              project=project, colour=colour)
    
    




def get_subj_IDs(x, y):
    """
    

    Returns
    -------
    subjs : list
        List containing all existing subject ID strings.

    """
    
    #import ch_disp_data as rep_data
    
    # get repeated site ch disp data
    data_frame = load_ch_disp_data( str(x) + "_" + str(y) )
    
    subjs = list(dict.fromkeys(data_frame['subject']))
    
    return subjs


def load_ch_disp_data(prefix):
    """Load locally cached channels displacement data
    
    Data loaded from the one_params CACHE DIR.
    
    Parameters
    ----------
    prefix : str
        Specify the PREFIX for the title to save CSV.  CSV will be titled
        '<prefix>_ch_disp_from_planned.csv'.  Recommend to use the trajectory
        insertion x,y coords in µm as prefix. e.g. '-2243_-2000' for repeated
        site.

    Returns
    -------
    data_frame : pandas DataFrame
        Dataframe containing: subject, eid, probe; ins_x, ins_y; chan_loc; 
        planned_orth_proj; dist.

    """
    
    #from ibllib.io import params - deprecated!  Access via one.alyx._par.as_dict()
    
    from pathlib import Path
    import pandas as pd
    from one.api import ONE
    one = ONE()
    
    par = one.alyx._par.as_dict()
    
    import pandas as pd
    from pathlib import Path
    
    PROBE_DATA_REL_PATH = Path('histology', 
                               'probe_data', 
                               prefix+'_ch_disp_from_planned.csv')
    
    path_probe_data = Path(par['CACHE_DIR']).joinpath(PROBE_DATA_REL_PATH)
    path_probe_data.parent.mkdir(exist_ok=True, parents=True)
    
    if path_probe_data.exists():
        data_frame = pd.read_csv( str(path_probe_data) )
        data_frame = data_frame.drop(columns=['Unnamed: 0']) # drop index col
        return data_frame
    else:
        return None



def get_subj_IDs_rep_site():
    return get_subj_IDs(-2243, -2000)


# if file is executed as main - plot and save all repeated site channels
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    ids = get_subj_IDs()
    
    if os.path.exists('plots') is False:
        os.mkdir('plots')
    
    for i in ids:
        plots = plot_subj_channels_rep_site(i)
        plots['cax'].savefig( str(Path('plots', i+'_coronal.svg')) )
        plots['sax'].savefig( str(Path('plots', i+'_sagittal.svg')) )


