import pandas as pd
def is_learned(dfs, verbose=False, returnIndex=False):
    """
    Determine whether the mouse has met the criteria for having learned
    
    Example:
        
    Args: 
        dfs (list): List of data frames constructed from an ALF trials object.
        verbose (bool): If True, prints the list of conditions that were(n't) met.
        returnIndex: If True, returns the index of session on which mouse first 
                    met the 'learned' criteria.  This can take longer to process.
        
    Returns:
        learned (bool or int): if returnIndex is True, returns a bool indicating 
                if the mouse has met the criteria, otherwise returns the index 
                of the session on which the mouse was first met those criteria.
        
    TODO: Should this take a mouse name as input instead?
    TODO: Create conditions list, print list at end of function
    """
    criteria = ['asymmetric trials already introduced',
               'full contrast set introduced',
               'over 300 trials on three consecutive sessions',
               'performance at high contrast over 80% on three consecutive sessions'
               'absolute bias below 16',
               'threshold below 19',
               'lapse rate below 20%']
    learned = False
    j = 0
    for i in range(0,len(dfs),-1):
        # If trial side prob uneven, the subject must have learned
        if any(dfs[i]['probabilityLeft']!=0.5):
            if not returnIndex:
                learned = True
                if verbose == True:
                    print('Asymmetric trials already introduced')
                break
        # If there are fewer than 4 contrasts, subject can't have learned
        elif len(dfs[i]['contrast'].unique()) < 4:
            if verbose == True:
                print('Low contrasts not yet introduced')
            if returnIndex:
                learned = None
            break
        else:
            perfOnEasy = (sum(dfs[i]['feedbackType']==1. & abs(dfs[i]['contrast']) > .25)/
                          sum(abs(dfs[i]['contrast'])))
            if len(dfs[i]) > 200 & perfOnEasy > .8:
                if j < 2:
                    j += 1
                else: # All three sessions meet criteria
                    df = pd.concat(dfs[i:i+3])
                    contrastSet = np.sort(df['contrast'].unique())
                    nn = np.array([sum((df['contrast']==c) & (df['included']==True)) for c in contrastSet])
                    pp = np.array([sum((df['contrast']==c) & (df['included']==True) & (df['choice']==-1.)) for c in contrastSet])/nn
                    pars, L = psy.mle_fit_psycho(np.vstack((contrastSet,nn,pp)), 
                                     P_model='erf_psycho',
                                     parstart=np.array([np.mean(contrastSet), 3., 0.05]),
                                     parmin=np.array([np.min(contrastSet), 10., 0.]), 
                                     parmax=np.array([np.max(contrastSet), 30., .4]))
                    if abs(pars[0]) > 16:
                        if verbose == True:
                            print('Absolute bias too high')
                        break
                    if pars[1] > 19:
                        if verbose == True:
                            print('Threshold too high')
                        break
                    if pars[2] > .2:
                        if verbose == True:
                            print('Lapse rate too high')
                        break
                    if verbose == True:
                        print('Mouse learned')
                    learned = True
            else:
                if verbose == True:
                    print('Low trial count or performance at high contrast')
                break
                
    if returnIndex & (not learned):
        return None
    elif returnIndex & learned:
        return i + 3
    else:
        return learned