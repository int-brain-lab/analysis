

from oneibl.one import ONE
from plot import gen_figures

# if you could also try just adding 'ibllib - brainbox', 'iblscripts - certification', and 'analysis - cert_master_fn' repositories (on those branches) to your python path
import sys
sys.path.append('~/Documents/code/ibllib')
sys.path.append('~/Documents/code/iblscripts')

one = ONE()
eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
one.load(eid, dataset_types=one.list(), clobber=False, download_only=True)
gen_figures(eid, probe='probe_right', cluster_ids_summary=1)
