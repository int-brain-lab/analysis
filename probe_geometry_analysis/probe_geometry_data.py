#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jun  1 09:11:26 2020

Probe Geometry Data Collection

Functions to gather data on probe geometry from IBL servers


Following functions are defined:

* download_ : downloads the current data from IBL servers

* save_ : saves a dataframe, using specified prefix, to local cache.

* load_ : Load a dataframe, using a specified prefix, from local cache.


The following data is processed:

* channels : all channel locations and their orthogonal coord to the probes planned trajectory

* probe_trajectory : brain surface insertion coord and the probe tip coordinate.



Gather channels displacement data from planned traj at repeated site.

Following data is gathered:

* subject, eid, probe - the subject, eid and probe IDs

* chan_loc - xyz coord of channels

* planned_orth_proj - xyz coord of orthogonal line from chan_loc to planned proj

* dist - the 3D distance between chan_loc xyz and planned_orth_proj xyz


Three functions:
    
    download : download the data from Alyx, returns as pandas DataFrame
    
    load  : open a cached CSV which exists next to this file.

    save  : save a pandas dataframe to a cached location


If run as main: data is saved to CSV next to this module.

NB: Channels are derived from the serializer: histology processed, histology,
micro manipulator, planned - the first is always returned.

@author: stevenwest
'''


def download_channels_data(x, y, project='ibl_neuropixel_brainwide_01'):
    """Download channels geometry data for all given probes in a given project
    at the planned insertion coord [x,y] from Alyx.
    
    Downloads the most up-to-date data from Alyx for all recordings planned at 
    [x,y], including their channel positions, and the orthogonal coords of 
    each channel from the planned trajectory.
    
    Saves this data to a standard location in the file system.
    
    Also returns this data as a pandas DataFrame object with following cols:
    
    * subject, eid, probe - the subject, eid and probe IDs
    * chan_loc - xyz coords of all channels
    * planned_orth_proj - xyz coord of orthogonal line from chan_loc onto planned proj
    * dist - the euclidean distance between chan_loc xyz and planned_orth_proj xyz
    
    Parameters
    ----------
    x : int
        x planned insertion coord in µm.  Eg. repeated site is -2243
    y : int
        y planned insertion coord in µm. Eg. repeated site is -2000.
    project : str, optional
        Project to gather all trajectories from. The default is 
        'ibl_neuropixel_brainwide_01'.
    
    Returns
    -------
    data_frame : pandas DataFrame
        Dataframe containing: subject, eid, probe; ins_x, ins_y; chan_loc; 
        planned_orth_proj; dist.
    
    """
    
    from one.api import ONE
    from ibllib.atlas import Insertion
    import brainbox.io.one as bbone
    #from ibllib.io import params - deprecated!
    
    # for catching errors in collecting datasets
    from ibllib.exceptions import IblError
    from urllib.error import HTTPError
    
    import numpy as np
    import pandas as pd
    #from pathlib import Path
    
    # create prefix object: str in format -2000_-2243
        # used to save the resulting data to local cache!
    prefix = str(str(x)+"_"+str(y))
    
    # connect to ONE
    one = ONE()
    
    # get all trajectories with Planned provenance, at [x,y], in project
    traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         x=x, y=y,  project=project)
    
    
    # from this collect all eids, probes, subjects that have this traj
    eids = [sess['session']['id'] for sess in traj]
    probes = [sess['probe_name'] for sess in traj]
    subj = [sess['session']['subject'] for sess in traj]
    
    
    # new dict to store data in loop:
     # chan_loc - xyz coord of channels
     # planned_orth_proj - xyz coord of orthogonal line from chan_loc to planned 
     #   proj
     # dist - the 3D distance between chan_loc xyz and planned_orth_proj xyz
    data = {
        
        'subject': [],
        'lab': [],
        'eid': [],
        'probe': [],
        
        'ins_x': [],
        'ins_y': [],
        
        'chan_loc_x': [],
        'chan_loc_y': [],
        'chan_loc_z': [],
        
        'planned_orth_proj_x': [],
        'planned_orth_proj_y': [],
        'planned_orth_proj_z': [],
        
        'dist': [],
    
    }
    
    
    # Fetch Repeated Site planned trajectory metadata:
    # Get the planned trajectory metadata
    planned = one.alyx.rest('trajectories', 'list', session=eids[0],
                 probe=probes[0], provenance='planned')
            
    # create insertion object of probe from planned trajectory:
    ins = Insertion.from_dict(planned[0])
            
    # create a trajectory object of Planned Repeated Site from this insertion:
    traj = ins.trajectory
    
    subindex=0
    
    # loop through each eid/probe:
    for eid, probe in zip(eids, probes):
        
        print("==================================================================")
        print(eids.index(eid))
        print(eid)
        print(probe)
        print(subj[subindex])
        subindex=subindex+1
    
        # get the eid/probe as insertion
        insertion = one.alyx.rest('insertions', 'list', session=eid, 
                              name=probe)
        
        if insertion:
            
            print("  insertion exists")
            
            # check if histology has been traced and loaded
            tracing = np.array(insertion[0]['json'])
            
            if tracing:
                
                print("  tracing exists")
                
                # For this insertion which has histology tracing, retrieve the
                # channels in xyz coords:
                
                # check the localCoordinates EXIST for this eid/probe
                    # run in a try..except statement to continue over the eid/probe
                    # if localCoordinates dataset does not exist
                try:
                    channel_coord = one.load_dataset(
                        eid,
                        'channels.localCoordinates.npy', 
                        collection='alf/'+probe)
                except IblError:
                    print("ALFObjectNotFound")
                    print("")
                    continue
                except HTTPError:
                    print("HTTPError")
                    print("")
                    continue
                except:
                    print("ERROR - generic")
                    continue
                
                
                # only proceed if channel_coord is not None
                if channel_coord is None:
                    continue
                
                print("  channel_coords exist")
                
                if one.alyx.rest('trajectories', 'list', session=eid, probe=probe,
                                        provenance='Histology track') == []:
                    print("ERROR - no Histology Track..")
                    continue
                
                chan_loc = bbone.load_channel_locations(eid, one=one, probe=probe)
                
                print("chan_loc")
                
                # only proceed if channel locations could be retrieved
                if not chan_loc:
                    continue
                
                # Next, create a representation of the planned trajectory as a
                # line:
                plannedTraj = one.alyx.rest('trajectories', 'list', session=eid,
                                        probe=probe, provenance='planned')
                
                print("plannedTraj")
            
                # create insertion object from planned trajectory:
                #ins = Insertion.from_dict(planned[0])
            
                # create a trajectory object from this insertion:
                #traj = ins.trajectory
            
            
                # NEXT - compute the projected coord for each channel coord onto the
                # line defined by traj:
                for ch_ind in range(len(chan_loc[probe]['x'])):
                
                    cl_x = chan_loc[probe]['x'][ch_ind]
                    cl_y = chan_loc[probe]['y'][ch_ind]
                    cl_z = chan_loc[probe]['z'][ch_ind]
                    
                    # create numpy array from chan_loc coords:
                    ch_loc = np.array([cl_x, cl_y, cl_z])
                
                    # project the current chan_loc to the PLANNED trajectory:
                    proj = traj.project(ch_loc)
                
                    # calculate the distance between proj and chan_loc:
                    dist = np.linalg.norm( ch_loc - proj )
                    
                    data['subject'].append(plannedTraj[0]['session']['subject'])
                    data['lab'].append(plannedTraj[0]['session']['lab'])
                    data['eid'].append(eid)
                    data['probe'].append(probe)
                    data['ins_x'].append(probe)
                    data['ins_y'].append(probe)
                    data['chan_loc_x'].append(cl_x)
                    data['chan_loc_y'].append(cl_y)
                    data['chan_loc_z'].append(cl_z)
                    data['planned_orth_proj_x'].append(proj[0])
                    data['planned_orth_proj_y'].append(proj[1])
                    data['planned_orth_proj_z'].append(proj[2])
                    data['dist'].append(dist)
    
    
    # convert data to a Pandas DataFrame:
    data_frame = pd.DataFrame.from_dict(data)
    
    save_channels_data(data_frame, prefix)
    
    return data_frame



def download_channels_data_rep_site():
    """Download REPEATED SITE channels displacement data from Alyx
    
    CONVENIENCE FUNCTION - calls download_cnalles_data(-2243, -2000)
    
    Downloads the most up-to-date data from Alyx for all repeated site 
    recordings, including their channel positionss, and the orthogonal points 
    of each channel from the planned trajectory.
    
    Saves this data to a standard location in the file system.
    
    Also returns this data as a pandas DataFrame object with following:
    
    * subject, eid, probe - the subject, eid and probe IDs
    * chan_loc - xyz coord of channels
    * planned_orth_proj - xyz coord of orthogonal line from chan_loc to planned proj
    * dist - the 3D distance between chan_loc xyz and planned_orth_proj xyz
    
    Returns
    -------
    data_frame : pandas DataFrame
        Dataframe containing: subject, eid, probe; ins_x, ins_y; chan_loc; 
        planned_orth_proj; dist.

    """
    # call download_ch_disp_from_planned_data with rep site coords
    return download_channels_data(-2243, -2000)



def save_channels_data(data_frame, prefix):
    """Save channels displacement data to local cache
    
    Saves data inside the one_params CACHE DIR.
    
    Parameters
    ----------
    data_frame : pandas DataFrame
        DataFrame containing data to save.
        
    prefix : str
        Specify the PREFIX for the title to save CSV.  CSV will be titled
        '<prefix>_ch_disp_from_planned.csv'.  Recommend to use the trajectory
        insertion x,y coords in µm as prefix. e.g. '-2243_-2000' for repeated
        site.
    
    Returns
    -------
    None.

    """
    
    #from ibllib.io import params - deprecated!  Access via one.alyx._par.as_dict()
    
    from pathlib import Path
    from one.api import ONE
    one = ONE()
    # get alyx parameters from local system
    par = one.alyx._par.as_dict()
    
    # define the sub-path within the CACHE DIR
    CHANNELS_DATA_REL_PATH = Path('histology', 
                               'probe_data', 
                               prefix+'_channels_data.csv')
    
    # define full path - CACHE_DIR plus sub path
    path_channels_data = Path(par['CACHE_DIR']).joinpath(CHANNELS_DATA_REL_PATH)
    
    path_channels_data.parent.mkdir(exist_ok=True, parents=True)
    print("Written parent DIR: ", path_channels_data.parent)
    
    data_frame.to_csv( str(path_channels_data) )
    print("Written CSV file: ", path_channels_data)
    



def load_channels_data(prefix):
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
    
    # define the sub-path within the CACHE DIR
    CHANNELS_DATA_REL_PATH = Path('histology', 
                               'probe_data', 
                               prefix+'_channels_data.csv')
    
    # define full path - CACHE_DIR plus sub path
    path_channels_data = Path(par['CACHE_DIR']).joinpath(CHANNELS_DATA_REL_PATH)
    
    path_channels_data.parent.mkdir(exist_ok=True, parents=True)
    
    if path_channels_data.exists():
        data_frame = pd.read_csv( str(path_channels_data) )
        data_frame = data_frame.drop(columns=['Unnamed: 0']) # drop index col
        return data_frame
    else:
        return None



def load_channels_data_rep_site():
    """Load REPEATED SITE locally cached channels displacement data
    
    CONVENIENCE FUNCTION : calls load_channels_data("-2243_-2000")
    
    Data loaded from the one_params CACHE DIR.

    Returns
    -------
    data_frame : pandas DataFrame
        Dataframe containing: subject, eid, probe; ins_x, ins_y; chan_loc; 
        planned_orth_proj; dist.

    """
    return load_channels_data("-2243_-2000")



def download_probe_trajectory_data(x, y, 
                             project='ibl_neuropixel_brainwide_01'):
    """Download probe trajectory geometry data for all given probes in a given
    project at the planned insertion coord [x,y] from Alyx.
    
    Downloads the most up-to-date data from Alyx for all recordings at [x,y]
    
    Saves this data to a standard location in the file system: one params CACHE_DIR
    
    Also returns this data as a pandas DataFrame object with following:
    
    * subject lab eid probe - IDs

    * recording_date
    
    * planned_[x y z] planned_[theta depth phi]
    
    * micro_[x y z] micro_[theta depth phi]
    
    * micro_error_surf micro_error_tip
    
    * hist_[x y z] hist_[theta depth phi]
    
    * hist_error_surf hist_error_tip
    
    * hist_to_micro_error_surf hist_to_micro_error_tip
    
    * hist_ + micro_ : saggital_angle + _coronal_angle
    
    * mouse_recording_weight dob
    
    Parameters
    ----------
    x : int
        x insertion coord in µm.  Eg. repeated site is -2243
    y : int
        y insertion coord in µm. Eg. repeated site is -2000.
    project : str, optional
        Trajectory project to list from. The default is 
        'ibl_neuropixel_brainwide_01'.

    Returns
    -------
    data_frame : pandas DataFrame
        Dataframe containing retrieved data

    """
    
    from one.api import ONE
    import ibllib.atlas as atlas
    
    import numpy as np
    import pandas as pd
    
    #from datetime import date
    
    # in format -2000_-2243 - added to saved file name
    prefix = str(str(x)+"_"+str(y))
    
    
    # connect to ONE
    one = ONE()
    
    # get the planned trajectory for site [x,y]
    traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         x=x, y=y,  project=project)
    
    
    # from this collect all eids, probes, subjects, labs from traj
    eids = [sess['session']['id'] for sess in traj]
    probes = [sess['probe_name'] for sess in traj]
    subj = [sess['session']['subject'] for sess in traj]
    
    # get planned insertion - can get from any retrieved above in traj
    ins_plan = atlas.Insertion.from_dict(traj[0])
    
    # create a trajectory object of Planned Repeated Site from this insertion:
    #traj_plan = ins_plan.trajectory
    
    # new dict to store data from loop:
     # subject lab eid probe - IDs
     # recording_data - date of recording of the probe
     # planned micro hist - xyz theta/depth/phy
      # gives the insertion data xyz brain surface insertion plus angle and length
     # error surf/tip
      # euclidean dist at brain surface or tip between planned and micro/hist 
       # or micro and hist

    data = {
        
        'subject': [],
        'lab': [],
        'eid': [],
        'probe': [],

        'recording_date': [],

        'planned_x': [],
        'planned_y': [],
        'planned_z': [],
        'planned_theta': [],
        'planned_depth': [],
        'planned_phi': [],

        'micro_x': [],
        'micro_y': [],
        'micro_z': [],
        'micro_theta': [],
        'micro_depth': [],
        'micro_phi': [],

        'micro_error_surf': [],
        'micro_error_tip': [],

        'hist_x': [],
        'hist_y': [],
        'hist_z': [],
        'hist_theta': [],
        'hist_depth': [],
        'hist_phi': [],

        'hist_error_surf': [],
        'hist_error_tip': [],
        
        'hist_to_micro_error_surf': [],
        'hist_to_micro_error_tip': []

    }
    
    # get new atlas generating histology insertion
    brain_atlas = atlas.AllenAtlas(res_um=25)
    
    # loop through each eid/probe:
    for eid, probe in zip(eids, probes):
        
        print(" ")
        print(subj[eids.index(eid)])
        print("==================================================================")
        print(eids.index(eid))
        print(eid)
        print(probe)
    
        # get the eid/probe as insertion
        insertion = one.alyx.rest('insertions', 'list', session=eid, 
                              name=probe)
        
        print("insertion ")
        
        if insertion:
        
            # check if histology has been traced and loaded
            tracing = insertion[0]['json']
            
            if tracing is None:
                print("No tracing for this sample - skip")
                continue
            
            if "xyz_picks" not in tracing:
                print("No tracing for this sample - skip")
                continue
            
            print("tracing")
        
            if tracing:
                
                # For this insertion which has histology tracing, retrieve
                
                # CURRENT planned trajectory - to get subject and other metadata
                planned = one.alyx.rest('trajectories', 'list', session=eid,
                                    probe=probe, provenance='planned')
                
                # micro-manipulator trajectory and insertion
                micro_traj = one.alyx.rest('trajectories', 'list', session=eid,
                                  probe=probe, provenance='Micro-manipulator')
                micro_ins = atlas.Insertion.from_dict(micro_traj[0])
                
                print("micro_traj")
                
                # get histology trajectory and insertion
                 # this retrieves the histology traced track from xyz_picks
                track = np.array(insertion[0]['json']['xyz_picks']) / 1e6
                track_ins = atlas.Insertion.from_track(track, brain_atlas)
                #track_traj = track_ins.trajectory
                
                print("track_traj")
                
                # only proceed if micro_traj and track_ins is not None
                if micro_traj is None:
                    print("micro_traj is NONE - skip")
                    print("")
                    continue
                
                if track_ins is None:
                    print("track_ins is NONE - skip")
                    print("")
                    continue
                
                
                data['subject'].append(planned[0]['session']['subject'])
                data['lab'].append(planned[0]['session']['lab'])
                data['eid'].append(eid)
                data['probe'].append(probe)
                
                data['recording_date'].append(planned[0]['session']['start_time'][:10])
                
                data['planned_x'].append(planned[0]['x'])
                data['planned_y'].append(planned[0]['y'])
                data['planned_z'].append(planned[0]['z'])
                data['planned_depth'].append(planned[0]['depth'])
                data['planned_theta'].append(planned[0]['theta'])
                data['planned_phi'].append(planned[0]['phi'])
                
                data['micro_x'].append(micro_traj[0]['x'])
                data['micro_y'].append(micro_traj[0]['y'])
                data['micro_z'].append(micro_traj[0]['z'])
                data['micro_depth'].append(micro_traj[0]['depth'])
                data['micro_theta'].append(micro_traj[0]['theta'])
                data['micro_phi'].append(micro_traj[0]['phi'])
                
                # compute error from planned
                error = micro_ins.xyz[0, :] - ins_plan.xyz[0, :]
                data['micro_error_surf'].append(np.sqrt(np.sum(error ** 2) ) * 1e6)
                error = micro_ins.xyz[1, :] - ins_plan.xyz[1, :]
                data['micro_error_tip'].append(np.sqrt(np.sum(error ** 2) ) * 1e6)
                
                data['hist_x'].append(track_ins.x * 1e6)
                data['hist_y'].append(track_ins.y * 1e6)
                data['hist_z'].append(track_ins.z * 1e6)
                data['hist_depth'].append(track_ins.depth * 1e6)
                data['hist_theta'].append(track_ins.theta)
                data['hist_phi'].append(track_ins.phi)
                
                # compute error from planned
                error = track_ins.xyz[0, :] - ins_plan.xyz[0, :]
                data['hist_error_surf'].append(np.sqrt(np.sum(error ** 2) ) * 1e6)
                error = track_ins.xyz[1, :] - ins_plan.xyz[1, :]
                data['hist_error_tip'].append(np.sqrt(np.sum(error ** 2) ) * 1e6)
                
                # compute error from micro
                error = track_ins.xyz[0, :] - micro_ins.xyz[0, :]
                data['hist_to_micro_error_surf'].append(np.sqrt(np.sum(error ** 2) ) * 1e6)
                error = track_ins.xyz[1, :] - micro_ins.xyz[1, :]
                data['hist_to_micro_error_tip'].append(np.sqrt(np.sum(error ** 2) ) * 1e6)
    
        
    # HISTOLOGY DATA:
    # Using phi and theta calculate angle in SAGITTAL plane (beta)
    x = np.sin(np.array(data['hist_theta']) * np.pi / 180.) * \
        np.sin(np.array(data['hist_phi']) * np.pi / 180.)
    y = np.cos(np.array(data['hist_theta']) * np.pi / 180.)
    # add this data to the list:
    data['hist_saggital_angle'] = np.arctan2(x, y) * 180 / np.pi # hist_beta
    
    # Using phi and theta calculate angle in coronal plane (alpha)
    x = np.sin(np.array(data['hist_theta']) * np.pi / 180.) * \
        np.cos(np.array(data['hist_phi']) * np.pi / 180.)
    y = np.cos(np.array(data['hist_theta']) * np.pi / 180.)
    # add this data to the list:
    data['hist_coronal_angle'] = np.arctan2(x, y) * 180 / np.pi # hist_alpha
    
    
    # MICRO MANIPULATOR DATA:
    # Using phi and theta calculate angle in sagittal plane (beta)
    x = np.sin(np.array(data['micro_theta']) * np.pi / 180.) * \
        np.sin(np.array(data['micro_phi']) * np.pi / 180.)
    y = np.cos(np.array(data['micro_theta']) * np.pi / 180.)
    # add this data to the list:
    data['micro_saggital_angle'] = np.arctan2(x, y) * 180 / np.pi # micro_beta
    
    # Using phi and theta calculate angle in coronal plane (alpha)
    x = np.sin(np.array(data['micro_theta']) * np.pi / 180.) * \
        np.cos(np.array(data['micro_phi']) * np.pi / 180.)
    y = np.cos(np.array(data['micro_theta']) * np.pi / 180.)
    # add this data to the list:
    data['micro_coronal_angle'] = np.arctan2(x, y) * 180 / np.pi # micro_alpha
    
    
    # Get mouse weights around time of recordings
     # mouse weights from Alyx https://github.com/int-brain-lab/ibllib/issues/50
    rec_wts = [] # empty list to append weights to
    # rec_subjs = [] # also track the subject ID to check its correct!
     # it is correct to commented out :)
    
    for s,r in zip(data['subject'], data['recording_date']):
        wts = one.alyx.rest('subjects', 'read', s)
        print(s)
        for w in wts['weighings']:
            if w['date_time'][:10] == r:
                print(r)
                rec_wts.append(w['weight'])
                #rec_subjs.append(s)
                break # only add one weight per subject
    
    data['mouse_recording_weight'] = rec_wts
    #data['rec_subj'] = rec_subjs
    
    # get dobs
    dobs = []
    for s in data['subject']:
        subject_list = one.alyx.rest('subjects', 'list', nickname = s)
        dobs.append(subject_list[0]['birth_date'])
    
    data['dob'] = dobs
    
    # compute days alive at recording from dob and recording_date
    #age_days = []
    #for d in range(len(data)):
        #age_days.append((date( int(data['recording_date'][d][:4]), int(data['recording_date'][d][5:7]), int(data['recording_date'][d][8:10]) ) - date( int(data['dob'][d][:4]), int(data['dob'][d][5:7]), int(data['dob'][d][8:10]) ) ).days)
    
    
    # convert data to a Pandas DataFrame:
    data_frame = pd.DataFrame.from_dict(data)
    
    save_probe_trajectory_data(data_frame, prefix )
    
    return data_frame



def download_probe_trajectory_data_rep_site():
    """Download REPEATED SITE probe displacement data from Alyx
    
    Downloads the most up-to-date data from Alyx for all repeated site 
    recordings, including their channel positions, and the orthogonal points 
    of each channel from the planned trajectory.
    
    Saves this data to a standard location in the file system.
    
    Also returns this data as a pandas DataFrame object with following:
    
    Also returns this data as a pandas DataFrame object with following:
    
    * subject lab eid probe - IDs

    * recording_date
    
    * planned_[x y z] planned_[theta depth phi]
    
    * micro_[x y z] micro_[theta depth phi]
    
    * micro_error_surf micro_error_tip
    
    * hist_[x y z] hist_[theta depth phi]
    
    * hist_error_surf hist_error_tip
    
    * hist_to_micro_error_surf hist_to_micro_error_tip
    
    Returns
    -------
    data_frame : pandas DataFrame
        Dataframe containing probe displacemnt data

    """
    # call download_ch_disp_from_planned_data with rep site coords
    return download_probe_trajectory_data(-2243, -2000)



def save_probe_trajectory_data(data_frame, prefix):
    """Save Probe displacement data to local cache
    
    Saves data inside the one_params CACHE DIR.
    
    Parameters
    ----------
    data_frame : pandas DataFrame
        DataFrame containing data to save.
        
    prefix : str
        Specify the PREFIX for the title to save CSV.  CSV will be titled
        '<prefix>_probe_disp.csv'.  Recommend to use the trajectory
        insertion x,y coords in µm as prefix. e.g. '-2243_-2000' for repeated
        site.
    
    Returns
    -------
    None.

    """
    
    #from ibllib.io import params - deprecated!  Access via one.alyx._par.as_dict()
    
    from pathlib import Path
    from one.api import ONE
    one = ONE()
    
    par = one.alyx._par.as_dict()
    
    PROBE_DATA_REL_PATH = Path('histology', 
                               'probe_data', 
                               prefix+'_probe_disp.csv')
    
    path_probe_data = Path(par['CACHE_DIR']).joinpath(PROBE_DATA_REL_PATH)
    
    path_probe_data.parent.mkdir(exist_ok=True, parents=True)
    print("Written parent DIR: ", path_probe_data.parent)
    
    data_frame.to_csv( str(path_probe_data) )
    print("Written CSV file: ", path_probe_data)
    



def load_probe_trajectory_data(prefix, suffix='_probe_disp.csv'):
    """Load locally cached probe displacement data
    
    Data loaded from the one_params CACHE DIR.
    
    Parameters
    ----------
    prefix : str
        Specify the PREFIX for the title to save CSV.  CSV will be titled
        '<prefix>_probe_disp.csv'.  RECOMMEND to use the trajectory
        insertion x,y coords in µm as prefix. e.g. '-2243_-2000' for repeated
        site.
    
    suffix : str
        Set by default to '_probe_disp.cs', which is the default used by 
        save_probe_disp_data() function. RECOMMEND to use this for consistency!

    Returns
    -------
    data_frame : pandas DataFrame
        Dataframe containing probe disp data

    """
    
    #from ibllib.io import params - deprecated!  Access via one.alyx._par.as_dict()
    
    from pathlib import Path
    from one.api import ONE
    one = ONE()
    
    par = one.alyx._par.as_dict()
    
    import pandas as pd
    from pathlib import Path
    
    PROBE_DATA_REL_PATH = Path('histology', 
                               'probe_data', 
                               prefix + suffix )
    
    path_probe_data = Path(par['CACHE_DIR']).joinpath(PROBE_DATA_REL_PATH)
    path_probe_data.parent.mkdir(exist_ok=True, parents=True)
    
    if path_probe_data.exists():
        data_frame = pd.read_csv( str(path_probe_data) )
        data_frame = data_frame.drop(columns=['Unnamed: 0']) # drop index col
        return data_frame
    else:
        return None



def load_probe_trajectory_data_rep_site():
    """Load REPEATED SITE locally cached probe trajectory data
    
    Data loaded from the one_params CACHE DIR.

    Returns
    -------
    data_frame : pandas DataFrame
        Dataframe containing: subject, eid, probe; ins_x, ins_y; chan_loc; 
        planned_orth_proj; dist.

    """
    return load_probe_trajectory_data("-2243_-2000")




def get_subj_IDs(x, y):
    """Get Subject IDs that have probe geometries planned at [x,y]
    

    Returns
    -------
    subjs : list
        List containing all existing subject ID strings.

    """
    
    
    # get repeated site ch disp data
    data_frame = load_channels_data( str(x) + "_" + str(y) )
    
    subjs = list(dict.fromkeys(data_frame['subject']))
    
    return subjs




def get_subj_IDs_rep_site():
    """CONVENIENCE FUNCTION called get_subj_IDs(-2243, -2000)
    
    """
    return get_subj_IDs(-2243, -2000)


