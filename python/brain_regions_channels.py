from ibllib.pipes.ephys_alignment import EphysAlignment
from oneibl.one import ONE
from brainbox.io.one import load_channel_locations
import numpy as np
import matplotlib.pyplot as plt

one = ONE(base_url="https://alyx.internationalbrainlab.org")
# Load data from 'ZM_2407' '2019-12-06'
eid = '03d3cffc-755a-43db-a839-65d8359a7b93'
probe = 'probe_00'
channels = load_channel_locations(eid, one=one, probe=probe)
# xyz coords of channels
xyz_channels = np.c_[channels[probe].x, channels[probe].y, channels[probe].z]
# depth along probe of channels
depths = channels[probe].axial_um
region, region_label, region_colour, region_id = EphysAlignment.get_histology_regions(xyz_channels, depths)

fig, ax1 = plt.subplots(figsize=(4,10))
for reg, col in zip(region, region_colour):
    height = np.abs(reg[1]- reg[0])
    bottom = reg[0]
    color = col/255
    ax1.bar(x=0.5, height=height, width=1, color=color, bottom=reg[0], edgecolor='w')

ax1.set_yticks(region_label[:, 0].astype(int))
ax1.set_yticklabels(region_label[:, 1])
ax1.hlines([0, 3840], *ax1.get_xlim(), linestyles='dashed', linewidth = 3, colors='k')
plt.show()

