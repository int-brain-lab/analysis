# Gaelle Chapuis
# find mice in database, that are: 1) trained, 2) male or female, 3) belonging to specific lab,
# 4) belonging to specific project. If no argument is input, all trained mice are returned.
# ---
# Note: for a quick an dirty way to get a selection of mice, you can e.g. run:
# ---
# attest2 = subject.Subject * subject.SubjectProject * subject.SubjectLab
# select_mice = attest2 & 'sex="M"' & 'subject_project="ibl_neuropixel_brainwide_01"' &
# 'lab_name="mrsicflogellab"'

# %%
from ibl_pipeline import subject
from ibl_pipeline.analyses import behavior as behavior_analyses
# %%
SEX_DEFAULTS = ('m', 'M', 'f', 'F', 'mf')
TRAIN_DEFAULT = ('trained')  # todo find a way to get unique names from DJ


def get_mice(sex='mf', lab_name=None, training_status=None, format='frame',
             project_name=None):

    if not isinstance(training_status, str):
        raise ValueError('training_status has to be a string')
    # if not isinstance(project_name, str):
    #     raise ValueError('project_name has to be a string')
    if sex not in SEX_DEFAULTS:
        raise ValueError('if a specific sex is wanted, sex has to be written either m or f.')
    if training_status not in TRAIN_DEFAULT:
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

    # Project criterion
    if project_name is None:
        project = subject.SubjectProject
    else:
        project = subject.SubjectProject & (f'subject_project="{project_name}"')

    #  Training criterion
    if training_status is None:
        trained = behavior_analyses.SessionTrainingStatus
    else:
        trained = behavior_analyses.SessionTrainingStatus & (f'training_status="{training_status}"')

    #  Assemble multiple conditions
    subject_output = subject_sex & subject_lab & project & trained

    #  Fetch and return data
    return subject_output.fetch(format=format)
