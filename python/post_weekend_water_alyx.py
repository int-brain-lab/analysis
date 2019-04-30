"""Usage:

source activate ibllibenv
python post_weekend_water_alyx.py 'today'
python post_weekend_water_alyx.py 'tomorrow'
python post_weekend_water_alyx.py '3days'
python post_weekend_water_alyx.py '2019-04-30T13:00'

Anne Urai, CSHL, 2019
"""

from oneibl.one import ONE
import datetime
import pandas as pd
import sys
from IPython import embed as shell

# run this script to post adlib CA water for all CSHL mice on Friday, Saturday and Sunday
one = ONE(base_url='https://alyx.internationalbrainlab.org')
days = sys.argv[1]
print(days)

# based on input argument to Python, which dates to use?
# we can add more dates here, or use 'YY-MM-DDTHH:MM'
if days == 'today':
	dates = [datetime.datetime.today().strftime('%Y-%m-%dT%H:%M')]
elif days == 'tomorrow':
	dates = datetime.datetime.today() + datetime.timedelta(days=1)
	dates = [dates.strftime('%Y-%m-%dT%H:%M')]
elif days == '3days':
	today 	 = datetime.datetime.today().strftime('%Y-%m-%dT%H:%M')
	tomorrow = datetime.datetime.today() + datetime.timedelta(days=1)
	tomorrow = tomorrow.strftime('%Y-%m-%dT%H:%M')
	dayaftertomorrow = datetime.datetime.today() + datetime.timedelta(days=2)
	dayaftertomorrow = dayaftertomorrow.strftime('%Y-%m-%dT%H:%M')
	dates = [today, tomorrow, dayaftertomorrow]
else:
	# print('invalid date. correct syntax: python post_weekend_water_alyx.py ''today'', or ''2019-04-30T13:00''')
	dates = [days]

print(dates)
subjects = pd.DataFrame(one.alyx.get('/subjects?&alive=True&responsible_user=valeria'))
sub = subjects['nickname'].unique()

for dat in dates:
    for s in sub:
        # settings
        wa_ = {
            'subject': s,
            'date_time': dat,
            'water_type': 'Water 2% Citric Acid',
            'user': 'valeria',
            'adlib': True}
        userresponse = input('Post adlib Citric Acid Water to mouse %s on date %s? '%(s, dat))
        if userresponse.lower() == 'yes':
        	# post on Alyx
        	rep = one.alyx.rest('water-administrations', 'create', wa_)
        	print('POSTED adlib Citric Acid Water to mouse %s on date %s.'%(s, dat))
        else:
        	pass