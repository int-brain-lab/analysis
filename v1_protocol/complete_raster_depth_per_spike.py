from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from oneibl.one import ONE
import alf.io
import scipy.stats
plt.ion()

def scatter_raster(spikes, clusters=[], boundary_times=None, downsample_factor=25):
 
    '''
    Create a scatter plot, time vs depth for each spike
    colored by cluster id; including vertical lines
    for stimulus type boundary times

    Note that interval should be at most 10**6 else
    the plot is too memory expensive

    :param spike: spike = alf.io.load_object(alf_path, 'spikes')
    :type spike: dict
    :param boundary_times: start/end times of v1 cert stim types
    :type boundary_times: dict 
    :param downsample_factor: n, only every nth spike is kept 
    :type downsample_factor: int
    :param clusters: clusters that should be plot 
    :type clusters: list
    :rtype: plot
    '''    
     
    if not(clusters):
        print('All clsuters are shown')
        uclusters = np.unique(spikes['clusters'])
        # downsample 
        z = spikes['clusters'][::downsample_factor]
        x = spikes['times'][::downsample_factor]
        y = spikes['depths'][::downsample_factor]
    else:
        print('Only a subset of all clusters is shown')
        Mask = np.isin(spikes['clusters'], clusters)
 
        Clusters = spikes['clusters'][Mask]
        Times = spikes['times'][Mask]
        Depths = spikes['depths'][Mask]

        uclusters = np.unique(Clusters)
        # downsample 
        z = Clusters[::downsample_factor]
        x = Times[::downsample_factor]
        y = Depths[::downsample_factor]
   

    fig, ax = plt.subplots() 
    
    cols = ['c','b','g','y','k','r','m']
    cols_cat = (cols*int(len(uclusters)/len(cols)+10))[:len(uclusters)]
    col_dict = dict(zip(uclusters, cols_cat))
    cols_int =[col_dict[x] for x in z]

    plt.scatter(x, y, marker='o', s=0.01, c = cols_int)

    # add vertical lines indicating stimulus type changes
    if boundary_times != None:
        for i in boundary_times:
            plt.axvline(boundary_times[i][0], linestyle='--', c='k')
            plt.text(boundary_times[i][0]+0.1,0,i+', start',rotation=90)
#            plt.axvline(boundary_times[i][1], linestyle='--', c='r')
#            plt.text(boundary_times[i][1]+0.1,0,i+', end',rotation=90) 

    plt.ylabel('depth [um]')
    plt.xlabel('time [sec]')
    plt.title('downsample factor: %s' %downsample_factor)  
    plt.show()


def get_stimulus_type_boundary_times(alf_path):

    '''
    from the _iblcertif_*times* files in alf,
    create dictionary of stimuls type names
    and start/end times
    '''

    times = list(Path(alf_path).rglob('_iblcertif_*times*'))

    T ={}
    for t in times:
        name = '.'.join(str(t).split('/')[-1].split('.')[1:4])        
        T[name] = np.load(t)

    T2 = {}
    for i in T:
        if len(T[i].shape) == 2:
            T2[i] = [T[i][0][0], T[i][-1][-1]]
        if len(T[i].shape) == 1:
            T2[i] = [T[i][0], T[i][-1]]

    return T2


def scatter_with_boundary_times(eid, clusters=[]):

    one = ONE()
    #eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)
    #eid = one.search(subject='ZM_2407', date='2019-11-05', number=3) #depth per spike but no times
    D = one.load(eid[0], clobber=False, download_only=True)
    alf_path = Path(D.local_path[0]).parent
    T2 = get_stimulus_type_boundary_times(alf_path)
    spikes = alf.io.load_object(alf_path, 'spikes')
    scatter_raster(spikes, clusters=clusters, boundary_times=T2)



#one.load(eid, dataset_types=one.list(), clobber=False, download_only=True)

#if __name__ == '__main__':

#    one = ONE()
#    eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)
#    #eid = one.search(subject='ZM_2407', date='2019-11-05', number=3) #depth per spike but no times
#    D = one.load(eid[0], clobber=False, download_only=True)
#    alf_path = Path(D.local_path[0]).parent
#    T2 = get_stimulus_type_boundary_times(alf_path)
#    spikes = alf.io.load_object(alf_path, 'spikes')
#    scatter_raster(spikes, boundary_times=T2)


