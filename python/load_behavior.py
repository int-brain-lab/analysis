def load_behavior(ref, rootDir=None):
    """
    Load the trials for a given experiment reference
    
    Example:
        df = load_behaviour('2018-09-11_1_MOUSE', rootDir = r'\\server1\Subjects')
        df.head()
        
    Args: 
        subject (str): The subject name
        rootDir (str): The root directory, i.e. where the subject data are stored.
                       If rootDir is None, the current working directory is used.
        
    Returns:
        df (DataFrame): DataFrame constructed from the trials object of the ALF 
                        files located in the experiment directory
    
    TODO: return multi-level data frame
           
    @author: Miles
    """
    import pandas as pd
    from os import listdir, getcwd
    from os.path import isfile, join
    
    if rootDir is None:
        rootDir = getcwd()
    path = dat.exp_path(ref, rootDir)
    alfs = [f for f in listdir(path) if (isfile(join(path, f))) & (is_alf(f))]
    parts = [alf_parts(alf) for alf in alfs]
    # List of 'trials' attributes
    attr = [parts[i]['typ'] for i in range(len(parts)) if parts[i]['obj'] == 'trials']
    attr.extend(['trialStart', 'trialEnd'])
    # Pull paths of trials ALFs
    trials = [join(path,f) for f in alfs if 'trials' in f]
    if not trials:
        print('{}: Nothing to process'.format(ref))
        return
    # Load arrays into dictionary
    trialsDict = dict.fromkeys(attr)
    for p,name in zip(trials, attr):
        trialsDict[name] = np.load(p).squeeze()
    # Check all arrays the same length
    lengths = [len(val) for val in [trialsDict.values()]]
    assert len(set(lengths))==1,'Not all arrays in trials the same length'
    # Deal with intervals
    trialsDict['trialStart'] = trialsDict['intervals'][:,0]
    trialsDict['trialEnd'] = trialsDict['intervals'][:,1]
    trialsDict.pop('intervals', None)
    # Create data from from trials dict
    df = pd.DataFrame(trialsDict)
    df['contrast'] = (df['contrastRight']-df['contrastLeft'])*100
    df.name = ref
    return df