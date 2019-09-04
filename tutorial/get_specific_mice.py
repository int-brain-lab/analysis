# Gaelle Chapuis
# find mice in database, that are: 1) trained, 2) male or female, 3) belonging to specific lab


# %%
import datajoint as dj
from ibl_pipeline import reference, subject, action, acquisition, data, behavior, ephys
from ibl_pipeline.analyses import behavior as behavior_analyses
# %%
SEX_DEFAULTS = ('m', 'M', 'f', 'F', 'mf')


def get_trained_mice(sex='mf', lab_name=None, training_status='trained', format='frame'):

    if not isinstance(training_status, str):
        raise ValueError('training_status has to be a string')
    if sex not in SEX_DEFAULTS:
        raise ValueError('if a specific sex is wanted, sex has to be written either m or f.')

    #  Sex criterion
    if sex == 'mf':
        subject_sex = subject.Subject
    else:
        subject_sex = subject.Subject & {'sex': sex}

    # Lab criterion
    if lab_name is None:
        subject_lab = subject.SubjectLab
    else:
        subject_lab = subject.SubjectLab & {'lab_name': lab_name}

    #  Training criterion
    trained = behavior_analyses.SessionTrainingStatus & (f'training_status="{training_status}"')

    # Assemble multiple conditions
    subject_output = subject_sex & trained & subject_lab

    #  Fetch data
    return subject_output.fetch(format=format)
