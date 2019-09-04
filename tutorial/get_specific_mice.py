# Gaelle Chapuis
# find 10 mice from database, that are: 1) trained, 2) 5 female, 5 male


# %%
import datajoint as dj
from ibl_pipeline import reference, subject, action, acquisition, data, behavior, ephys
from ibl_pipeline.analyses import behavior as behavior_analyses
# %%
sex = 'm'
male = subject.Subject & {'sex': sex}
trained = behavior_analyses.SessionTrainingStatus & 'training_status="trained"'

male_trained = male & trained

subjs_mt = male_trained.fetch(format='frame')

n_subjs_mt = subjs_mt.shape[0]  # how many male trained mice there are in the database
# %%

if n_subjs_mt >= 5:  # get first 5 animals
    smt5 = subjs_mt.iloc[:5]
else:
    print('you do not have enough mice to run this analysis')
   


# attest2 = subject.Subject * subject.SubjectProject * subject.SubjectLab
# nummales_mf = len(attest2 & 'sex="M"' & 'subject_project="ibl_neuropixel_brainwide_01"' & 'lab_name="mrsicflogellab"')
# nummales_h = len(attest2 & 'sex="M"' & 'subject_project="ibl_neuropixel_brainwide_01"' & 'lab_name="hoferlab"')