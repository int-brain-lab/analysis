"""
Rename SfN files so biasedChoiceWorld extractor runs
Anne Urai, CSHL, 2019
"""

import os
import json
import glob

from shutil import copyfile

# use a rootdir on behavior rig 3, before transferring to server
rootdir = '/Users/urai/Downloads/Subjects/'
rootdir = os.path.join('Users', 'IBLuser', 'Downloads', 'SfN_humandata', 'Subjects')
os.chdir(rootdir)
print("Current Working Directory ", os.getcwd())

for file in glob.glob('**/_iblrig_taskSettings.raw.json', recursive=True):

    # save a backup copy
    copyfile(file, file.replace('.raw.', '.raw_backup.'))

    # do the actual replacement and save a new file
    with open(file) as f:
        s = f.read()
    s = s.replace('_iblrig_tasks_humanChoiceWorld', '_iblrig_tasks_human_biasedChoiceWorld')
    with open(file, "w") as f:
        f.write(s)
    print(file)

print('Done!')
