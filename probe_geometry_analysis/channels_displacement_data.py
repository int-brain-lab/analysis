#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jun  1 09:11:26 2020

Channels Displacement Data Collection

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


def download_ch_disp_data(x, y, 
                                       provenance='Planned', 
                                       project='ibl_neuropixel_brainwide_01'):
    """Download channels displacement data for a given probe at [x,y] from Alyx
    
    Downloads the most up-to-date data from Alyx for all recordings at [x,y],
    including their channel positionss, and the orthogonal points of each 
    channel from the planned trajectory.
    
    Saves this data to a standard location in the file system.
    
    Also returns this data as a pandas DataFrame object with following:
    
    * subject, eid, probe - the subject, eid and probe IDs
    * chan_loc - xyz coord of channels
    * planned_orth_proj - xyz coord of orthogonal line from chan_loc to planned proj
    * dist - the 3D distance between chan_loc xyz and planned_orth_proj xyz
    
    Parameters
    ----------
    x : int
        x insertion coord in µm.  Eg. repeated site is -2243
    y : int
        y insertion coord in µm. Eg. repeated site is -2000.
    provenance : str, optional
        Trajectory provenance to list from: Planned, Micro-manipulator, 
        Histology, E-phys aligned. The default is 'Planned'.
    project : str, optional
        Trajectory project to list from. The default is 
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
    from pathlib import Path
    
    # in format -2000_-2243
    prefix = str(str(x)+"_"+str(y))
    
    
    # connect to ONE
    one = ONE()
    
    # get the planned trajectory for repeated site: x=2243, y=2000
    traj = one.alyx.rest('trajectories', 'list', provenance=provenance,
                         x=x, y=y,  project=project)
    
    
    # from this collect all eids, probes, subjects that use repeated site:
    eids = [sess['session']['id'] for sess in traj]
    probes = [sess['probe_name'] for sess in traj]
    subj = [sess['session']['subject'] for sess in traj]
    
    
    # new dict to store data from loop:
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
        
        print("insertion ")
        
        if insertion:
        
            # check if histology has been traced and loaded
            tracing = np.array(insertion[0]['json'])
            
            print("tracing")
        
            if tracing:
                
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
                
                print("channel_coords")
                
                # only proceed if channel_coord is not None
                if channel_coord is None:
                    continue
                
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
    
    save_ch_disp_data(data_frame, prefix )
    
    return data_frame



def download_ch_disp_data_rep_site():
    """Download REPEATED SITE channels displacement data from Alyx
    
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
    return download_ch_disp_data(-2243, -2000)



def save_ch_disp_data(data_frame, prefix):
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
    
    par = one.alyx._par.as_dict()
    
    PROBE_DATA_REL_PATH = Path('histology', 
                               'probe_data', 
                               prefix+'_ch_disp_from_planned.csv')
    
    path_probe_data = Path(par['CACHE_DIR']).joinpath(PROBE_DATA_REL_PATH)
    
    path_probe_data.parent.mkdir(exist_ok=True, parents=True)
    print("Written parent DIR: ", path_probe_data.parent)
    
    data_frame.to_csv( str(path_probe_data) )
    print("Written CSV file: ", path_probe_data)
    



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



def load_ch_disp_data_rep_site():
    """Load REPEATED SITE locally cached channels displacement data
    
    Data loaded from the one_params CACHE DIR.

    Returns
    -------
    data_frame : pandas DataFrame
        Dataframe containing: subject, eid, probe; ins_x, ins_y; chan_loc; 
        planned_orth_proj; dist.

    """
    return load_ch_disp_data("-2243_-2000")



# if file is executed as main - fetch repeated site data and save cache as CSV
#if __name__ == "__main__":
#       data_frame = download_ch_disp_data_rep_site()



