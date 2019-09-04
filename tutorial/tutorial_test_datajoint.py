
# import wrappers etc
import datajoint as dj
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
from ibl_pipeline.utils import psychofit as psy
from ibl_pipeline.analyses import behavior as behavioral_analyses


import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# INITIALIZE A FEW THINGS
figpath = os.path.join(os.path.expanduser('~'), 'Data', 'Figures_IBL')
print(figpath)
print('datajoint rules!')