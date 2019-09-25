#!/usr/bin/env python

import numpy as np
import pandas as pd
import re

from os import listdir
from os.path import isfile, isdir


def getAllData_IBL(sourceDir, mouseIdx, **kwargs):
    '''
    03/20/2018 NAR

    Get data from all sessions and all trials for specified mouse
    
    Args:
        sourceDir : path to directory with raw data
        mouseIdx : index of desired rodent in the list mouseIDs
        **kwargs : will pass any keyword arguments received to unpackData_Raw()
            
    Returns:
        outData : data for the chosen mouse across all days and task conditions
    '''

    ### Initialize - prepare mouse dataset if no saved file, separated by mouse
    mouseData,mouseIDs = unpackData_Raw(sourceDir,**kwargs)

    ### Load specified mouse
    if type(mouseIdx) is int:
        myMouse = str(mouseIDs[mouseIdx])
    elif type(mouseIdx) is str:
        myMouse = mouseIdx

    try:
        outData = mouseData[myMouse]
        print('Mouse', myMouse, 'chosen for analysis...')
    except:
        raise Exception("Mouse with name " + myMouse + " could not be loaded, only mice " + str(mouseIDs))
    
    return outData



def unpackData_Raw(sourceDir, labs="All",
                   addHistory=2, addReward=2, addChoice=2, addStimuli=2, addBothStimuli=2,
                   scaleStimulus=True, forceNew=False, cutoff=50,
                   returnCount=False, verbose=False):
    '''
    03/22/2018 NAR

    Load data from IBL .npy files
    
    Args:
        sourceDir : str, file directory where all .npy files with data live
        labs : str or list, names of labs from which to process data
        addHistory : int, adds that many trials of history data (the correct
            choice on the previous trial(s)) to the current trial data
        addReward : int, adds that many trials of reward data (whether there
            was a reward on the previous trial(s)) to the current trial data
        addChoice : int, adds that many trials of choice data (which side the
            animal chose on the previous trial(s)) to the current trial data
        addStimuli : int, adds that many trials of stimuli data to current
            trial data
        addStimAvg : int, adds that many trials of stimuli data to current
            trial data, where both Left and Right stimuli are encoded together
        scaleStimulus : bool, normalizes the stimulus set
        forceNew : bool, will overlook existence of old processed data and
            force reprocessing
        cutoff : int, sessions with fewer valid trials will be removed
        returnCount : returns total number of mice in dataset instead of
            full dataset

    Returns:
        data : dictionary of dictionaries for each mouse
        mouseIDs : list of IDs for mice processed
        ~count : number of mice in dataset, returned when returnCount=True
    '''
    
    ### Process by lab analysis and specify output file name
    ALL_LABS = ['angelakilab', 'mainenlab', 'churchlandlab', 'wittenlab']
    if labs == 'All':
        output_filename = 'IBL_processed_data.npz'
        labs = ALL_LABS
    elif labs in ALL_LABS or labs=="mylab":
        output_filename = 'IBL_processed_data_' + labs + '.npz'
        labs = [labs]
    else:
        raise Exception("if labs is not 'All', must be one of" + str(ALL_LABS))

    output_file = sourceDir + output_filename


    ### Check to see if file exists, and if so, if reprocessing is forced
    ### Otherwise, simply return already processed data
    if isfile(output_file):

        if verbose: print("Raw data has already been processed")

        if forceNew:
            if verbose: print("Forcing reprocessing...")
        else:
            if verbose: print("Loading", output_file)
            with np.load(output_file) as oldOutput:
                data = oldOutput['data'].item()
                mouseNames = oldOutput['mouseNames']

            if returnCount: return len(data)
            else: return data, mouseNames


    ### If original raw files do not exist, raise error
    if not isdir(sourceDir):
        raise Exception('Raw file directory not found at:' + sourceDir)

    ### Read in responses
    req_extractVariables = ['_ibl_trials.feedbackType.npy', '_ibl_trials.choice.npy',
                        '_ibl_trials.contrastLeft.npy', '_ibl_trials.contrastRight.npy']
    req_extractNames = ['reward','choice','contrastLeft','contrastRight']

    aux_extractVariables = ['_ibl_trials.included.npy', '_ibl_trials.probabilityLeft.npy']
    aux_extractNames = ['include','probL']  
    
    trackedVariables = ['lab','mouseName','date','session']

    mouseData = pd.DataFrame(columns = req_extractNames + aux_extractNames + trackedVariables)

    for L in labs:
        sourceDir_lab = sourceDir + L + '/Subjects/'
        mouseNames = [i for i in listdir(sourceDir_lab) if isdir(sourceDir_lab+i)]

        for mN in mouseNames:

            if verbose: print("mouse", mN)
            trainDays = listdir(sourceDir_lab + mN)
            trainDays.sort()

            for tD in trainDays:

                if not isdir(sourceDir_lab + mN + '/' + tD): continue
                if verbose: print("   day", tD)
                numSession = listdir(sourceDir_lab + mN + '/' + tD)
                numSession.sort()

                for nS in numSession:

                    current_dir = sourceDir_lab + mN + '/' + tD + '/' + nS + '/alf/'
                    if not isdir(current_dir): continue
                    if verbose: print("      session", nS)

                    data_files = listdir(current_dir)

                    try:
                        myVars = {}
                        for ind in range(len(req_extractVariables)):

                            file_template = re.compile(req_extractVariables[ind][:-3] + '(.*?).npy')
                            name_data =  [file_template.findall(f) for f in data_files if len(file_template.findall(f))][0][0]

                            myVars[req_extractNames[ind]] = list(np.load(current_dir + '/' + req_extractVariables[ind][:-3] + name_data + ".npy").flatten())

                        for ind in range(len(aux_extractVariables)):
                            file_template = re.compile(aux_extractVariables[ind][:-3] + '(.*?).npy')

                            try:
                                name_data =  [file_template.findall(f) for f in data_files if len(file_template.findall(f))][0][0]
                                myVars[aux_extractNames[ind]] = list(np.load(current_dir + '/' + aux_extractVariables[ind][:-3] + name_data + ".npy").flatten())
                            except: pass

                        sessLen = len(myVars[req_extractNames[0]])
                        for key in myVars.keys():
                            myVars[key] = myVars[key][:sessLen]
                        myVars['lab'] = [L]*sessLen
                        myVars['mouseName'] = [mN]*sessLen
                        myVars['date'] = [tD]*sessLen
                        myVars['session'] = [nS]*sessLen

                        df = pd.DataFrame(myVars, columns=myVars.keys())
                        mouseData = mouseData.append(df, sort=True)

                    except:
                        if verbose: print("extractVariables for lab", L,
                            "for mouse", mN, "on day", tD, "in session", nS,
                            "were not loaded")
                        raise Exception()
 

    ### Separate by mouse
    data = {}
    mouseNames = mouseData.mouseName.unique()
    for mN in mouseNames:

        if verbose: print("Processing mouse:", mN)

        ### Removes dates occurring after the end of the automated training
        inputs = {}
        myMouse = mouseData[mouseData['mouseName'] == mN].copy()


        ### Drop trials from sessions of less than 'cutoff' valid trials
        myMouse = myMouse.groupby(['date','session']).filter(lambda x: len(x) >= cutoff)


        ### Drop any no-go trials
        myMouse = myMouse[myMouse['choice'] != 0].copy()


        ### Choice
        ### NOTE: CCW = -1 --> "Right Contrast" = +1 ; CW = +1 ---> "Left Contrast" = -1
        myMouse['choice'] = np.array(-myMouse['choice']/2 + 1.5).astype(int)
        if np.sum(~np.in1d(myMouse['choice'], [1,2])): raise Exception('invalid value for choice, not 1 or 2')

        ### Reward
        myMouse['reward'] = ((np.array(myMouse['reward'])+1)/2).astype(int)
        if np.sum(~np.in1d(myMouse['reward'], [0,1])): raise Exception('invalid value for reward, not 0 or 1')

        ### Answer --- Is reward synonymous with correct answer?
        myMouse['answer'] = (((-1)**myMouse['choice'] * -(-1)**myMouse['reward'])/2 + 1.5).astype(int)
        if np.sum(~np.in1d(myMouse['answer'], [1,2])): raise Exception('invalid value for answer, not 1 or 2')


        ### Get lab
        if len(myMouse['lab'].unique()) != 1: 
            print(myMouse['lab'].unique())
            raise Exception("mouse " + mN + " cannot come from multiple labs")
        lab = myMouse['lab'].unique()[0]


        ### Scale Stimulus
        if scaleStimulus:
            p = 3.5  # Set after manual search :/

            conL = np.abs(np.array(myMouse['contrastLeft']).copy())
            conR = np.abs(np.array(myMouse['contrastRight']).copy())
            conL[np.isnan(conL)] = 0
            conR[np.isnan(conR)] = 0
            myMouse['contrastLeft'] = conL
            myMouse['contrastRight'] = conR

            ### Raw contrast values tranformed by tanh
            myMouse['contrastLeft'] = np.tanh(p*myMouse['contrastLeft'])/np.tanh(p)
            myMouse['contrastRight'] = np.tanh(p*myMouse['contrastRight'])/np.tanh(p)


        ### For adding history variables, need to know
        dates = np.array(myMouse['date'])
        sessions = np.array(myMouse['session'])


        ### Add answer history variable(s) (this will not work if omitted trials are included)
        # {1 : right, -1 : left, 0 : no history}
        if addHistory:

            h = np.zeros((len(myMouse), addHistory))
            correct = np.array((-1) ** myMouse['answer'])

            for i in np.arange(addHistory):
                for j in np.arange(1+i,len(myMouse)):
                    if ((dates[j-i-1] == dates[j]) and
                        (sessions[j-i-1] == sessions[j])):
                        h[j, i] = correct[j-i-1] 
            
            inputs['h'] = h


        # Add reward history variable(s) (this will not work if omitted trials are included)
        # {1 : reward, -1 : no reward, 0 : no history}
        if addReward:

            r = np.zeros((len(myMouse), addReward))
            reward = np.array(-(-1) ** myMouse['reward'])

            for i in np.arange(addReward):
                for j in np.arange(1+i,len(myMouse)):
                    if ((dates[j-i-1] == dates[j]) and
                        (sessions[j-i-1] == sessions[j])):
                        r[j, i] = reward[j-i-1]
            
            inputs['r'] = r

 
        # Add choice history variable(s) (this will not work if omitted trials are included)
        # {1 : right, -1 : left, 0 : no history}
        if addChoice:

            c = np.zeros((len(myMouse), addChoice))
            choice = np.array((-1) ** myMouse['choice'])

            for i in np.arange(addChoice):
                for j in np.arange(1+i,len(myMouse)):
                    if ((dates[j-i-1] == dates[j]) and
                        (sessions[j-i-1] == sessions[j])):
                        c[j, i] = choice[j-i-1]

            inputs['c'] = c


        # Extend stimulus data back to previous trials
        if addStimuli:

            sL = np.zeros((len(myMouse), addStimuli))
            sR = np.zeros((len(myMouse), addStimuli))

            for i in np.arange(addStimuli):
                for j in np.arange(i,len(myMouse)):
                    if ((dates[j-i] == dates[j]) and
                        (sessions[j-i] == sessions[j])):
                        sL[j, i] = np.array(myMouse['contrastLeft'])[j-i]
                        sR[j, i] = np.array(myMouse['contrastRight'])[j-i]

            inputs['sL'] = sL
            inputs['sR'] = sR


        # Combine both left and right contrasts into one weight vector
        if addBothStimuli:

            sBoth = np.zeros((len(myMouse), addBothStimuli))

            for i in np.arange(addBothStimuli):
                for j in np.arange(i,len(myMouse)):
                    if ((dates[j-i] == dates[j]) and
                        (sessions[j-i] == sessions[j])):
                        sBoth[j, i] = np.array(myMouse['contrastLeft'])[j-i] - np.array(myMouse['contrastRight'])[j-i]
            
            inputs['sBoth'] = sBoth


        # Make sure all trial counts for each day start at 0
        sessLength = np.array(myMouse.groupby(['date','session']).size())


        ### Check for optional aux variables
        if 'include' in myMouse:
            repeat = np.array(myMouse['include']).astype(bool)
        else: repeat = None

        if 'probL' in myMouse:
            probL = np.array(myMouse['probL'])
        else: probL = None


        # Save dictionary of mouse specific info to larger dictionary under mouse's ID
        data[mN] = {
            
            # The name of the dataset
            'dataset' : "IBL_"+lab+"_Mouse",

            # The name of the animal
            'name' : mN,
            
            # A dict of the inputs received by the mouse on each trial
            # Each entry is a (*, N) array where each column i the input from i previous trials ago
            'inputs' : inputs,   

            # The choice made by the mouse {L:1, R:2, Omission:0}
            'y' : np.array(myMouse['choice']),

            # The correct answer of each trial {omission : 0, s1>s2 / L : 1, s2>s1 / R : 2}
            'answer' : np.array(myMouse['answer']),

            # Whether the mouse's chocie was correct on a trial {'incorrect' : 0, 'correct' : 1}
            'correct' : np.array(myMouse['reward']),

            # The date of each trial
            'date' : np.array(myMouse['date']),

            # The session number of each trial
            'session' : np.array(myMouse['session']).astype(int),

            # Array of how many trials the mouse had on each day of training, same length as dayList
            'dayLength' : sessLength,

            # Repeat trial or not
            'repeat' : repeat,

            # Indicate the current probability of presenting a leftward trial
            'probL' : probL,


            'conL' : conL,
            'conR' : conR
            }

    ### Save into file
    np.savez_compressed(output_file, data = data, mouseNames = mouseNames)
    
    if returnCount: return len(data)
    else:           return data, mouseNames