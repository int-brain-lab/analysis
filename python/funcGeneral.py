"""
Created on Wed Aug 22 16:08:04 2018
General behavioral functions
@author: Guido Meijer
"""

import os

def getPaths():
    root_path = '/media/guido/data/GDrive/IBL-Mainen/Rigbox_repository/'
    plot_path = '/home/guido/Projects/SteeringWheelBehavior/Plots/'
    return root_path, plot_path

def pullDrive(root_path, subject):
    # Pull data from Google drive (requires 'drive' installed)
    os.system('cd ' + root_path)
    os.system('drive pull IBL-Mainen/Rigbox_respository/' + subject +'/')

