"""
Import human data into DJ
Anne Urai, CSHL, 2019
With thanks to Steven Gluf for digitizing the data
"""

import pandas as pd
import datajoint as dj
from IPython import embed as shell # for debugging
import re

# import wrappers etc
from ibl_pipeline import subject, behavior

# =========================================================
# DEFINE THE SCHEMA
# =========================================================

schema = dj.schema('group_shared_sfndata')


@schema
class Subject(dj.Imported):
    definition = """
    -> subject.Subject
    ---
    subject_nickname:		        varchar(255)		# nickname
    sex=null:			            enum("M", "F", "U")	# sex
    age=null:                       int                 # age at data collection
    task_knowledge=null:            int                 # knowledge of the IBL task
    aq_score=null:                  int                 # Autism Quotient 10-question score
    """

    # only run this for subjects who did the task at SfN
    key_source = subject.Subject & 'subject_nickname LIKE "human%"'

    def make(self, key):

        # get nickname
        subject_nickname = (subject.Subject() & key).fetch1('subject_nickname')
        key['subject_nickname'] = subject_nickname
        subject_number = [int(s) for s in re.findall(r'\d+', subject_nickname)][0]

        # load data file
        data_file = pd.read_csv('~/Downloads/sfn_data_ibl.csv')

        # grab only the row that contains what we need
        thisdat = data_file.loc[data_file.subject_nickname == subject_number, :]

        if not thisdat.empty:
            # now insert
            if not thisdat.sex.isnull().values.any():
                key['sex'] = thisdat.sex.item()
            if not thisdat.age.isnull().values.any():
                key['age'] = int(thisdat.age.item())
            if not thisdat.knowledge.isnull().values.any():
                key['task_knowledge'] = thisdat.knowledge.item()
            if not thisdat.AQ.isnull().values.any():
                key['aq_score'] = thisdat.AQ.item()
        print(key)
        self.insert1(key)


# =================
# populate this
# =================
# Subject.drop()
Subject.populate(display_progress=True)
