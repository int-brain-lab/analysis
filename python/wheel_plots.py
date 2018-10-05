# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:39:52 2018

@author: Miles
"""

from os import listdir, getcwd
from os.path import isfile, join
from alf import is_alf, alf_parts

def load_wheel(ref, rootDir=None):
    """
    Load the wheel object for a given experiment reference
    
    Example:
        wheel, wm = load_wheel('2018-09-11_1_MOUSE', rootDir = r'\\server1\Subjects')
        wheel.head()
        
    Args: 
        subject (str): The subject name
        rootDir (str): The root directory, i.e. where the subject data are stored.
                       If rootDir is None, the current working directory is used.
        
    Returns:
        wheel (DataFrame): DataFrame constructed from the wheel object of the ALF 
                           files located in the experiment directory
        wm (DataFrame): DataFrame constructed from the wheelMoves object of the  
                           ALF files located in the experiment directory

    TODO: Deal with namespaces: currently hard-coded
    TODO: Make function more efficient: Does everything twice (once per ALF obj)
    TODO: Extract first few lines as decorator
    TODO: Move out of wheel_plots file
    """
    if rootDir is None:
        rootDir = getcwd()
    path = dat.exp_path(ref, rootDir)
    alfs = [f for f in listdir(path) if (isfile(join(path, f))) & (is_alf(f)) & (f.startswith('_ibl_wheel'))]
    if not alfs:
        print('{}: Nothing to process'.format(ref))
        return None, None
    # Pull paths of trials ALFs
    wheelPos = np.load(join(path, '_ibl_wheel.position.npy')).squeeze()
    wheelVel = np.load(join(path, '_ibl_wheel.velocity.npy')).squeeze()
    t = np.load(join(path, '_ibl_wheel.timestamps.npy')).squeeze()
    times = np.interp(np.arange(0,len(wheelPos)), t[:,0], t[:,1])
    wheel = pd.DataFrame({'position':wheelPos, 'velocity':wheelVel, 'times':times})
    
    intervals = np.load(join(path, '_ibl_wheelMoves.intervals.npy')).squeeze()
    try:
        movesType = pd.read_csv(join(path, '_ibl_wheelMoves.type.csv'), header=None)
        wm = pd.DataFrame({'onset':intervals[:,0], 'offset':intervals[:,1], 'type':movesType.values[0]})
    except: #TODO: Deal with missing movesType or empty file
        wm = None

    return wheel, wm
    
def plot_wheel_at_move_onset(wheelData, ax=None):
    """
    Plot the wheel traces for session aligned on movement onset.  The traces are
    coloured based on their classification. 
    
    Example:
        wheelData = load_wheel('2018-09-11_1_MOUSE', rootDir = r'\\server1\Subjects')
        plot_wheel_at_move_onset(wheelData)
        
    Args: 
        wheelData (tuple): A tuple containing a data frame of the wheel and wheel
                           movement ALF object data frames.
        
    Returns:
        ret (dict): A dictionary of the wheel movement types each holding a list
                    of line objects
        ax (Axes): The plot axes

    TODO: Deal with namespaces: currently hard-coded
    TODO: Make function more efficient: Does everything twice (once per ALF obj)
    TODO: Extract first few lines as decorator
    TODO: Move out of wheel_plots file
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.cla()
    wheel = wheelData[0]
    wm = wheelData[1]
    colours = {'CW':'b', 'CCW':'r', 'flinch':'k', 'other':'k'}
    ret = {'CW':[], 'CCW':[], 'flinch':[], 'other':[]}
    for i in range(0,len(wm)):
        t = (wheel['times'] > wm['onset'][i]) & (wheel['times'] < wm['offset'][i])
        pos = wheel['position'][t]
        wheelTimes = wheel['times'][t]
        relativeTimes = wheelTimes - wheelTimes.iloc[0]
        pos = pos - pos.iloc[0]
        ln, = ax.plot(relativeTimes*1000, pos.values, c=colours[wm['type'][i]], label=wm['type'][i])
        ret[wm['type'][i]].append(ln)
    ax.set_xlim([0, 10000])
    ax.set_ylim([-7, 7])
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Set bounds of axes lines
    #ax.spines['left'].set_bounds(0, 1)
    #ax.spines['bottom'].set_bounds(1, len(dfs)+1)
    # Explode out axes
    ax.spines['left'].set_position(('outward',10))
    ax.spines['bottom'].set_position(('outward',10))
    ax.set_xlabel('Time from movement onset (ms)')
    ax.set_ylabel('Relative position (cm)')
    return ret, ax
    
