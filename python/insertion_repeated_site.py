from oneibl.one import ONE
import ibllib.atlas as atlas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


one = ONE()

traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                     x=-2243, y=-2000,  project='ibl_neuropixel_brainwide_01')
eids = [sess['session']['id'] for sess in traj]
probes = [sess['probe_name'] for sess in traj]
subj = [sess['session']['subject'] for sess in traj]

data = {
    'subject': [],
    'eid': [],
    'probe': [],
    'planned_x': [],
    'planned_y': [],
    'planned_theta': [],
    'planned_depth': [],
    'planned_phi': [],
    'micro_x': [],
    'micro_y': [],
    'micro_theta': [],
    'micro_depth': [],
    'micro_phi': [],
    'hist_x': [],
    'hist_y': [],
    'hist_theta': [],
    'hist_depth': [],
    'hist_phi': []

}

brain_atlas = atlas.AllenAtlas(res_um=25)
d_types = ['probes.description']

# Collect all data for repeated site that has histology tracing
for eid, probe in zip(eids, probes):
    
    insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe)
    if insertion:
        tracing = np.array(insertion[0]['json'])
        if tracing:
            planned = one.alyx.rest('trajectories', 'list', session=eid,
                                    probe=probe, provenance='planned')
            micro = one.alyx.rest('trajectories', 'list', session=eid,
                                  probe=probe, provenance='Micro-manipulator')
            track = np.array(insertion[0]['json']['xyz_picks']) / 1e6
            track_coords = atlas.Insertion.from_track(track, brain_atlas)
            if not planned:
                planned = np.copy(micro)
            if not micro:
                micro = np.copy(planned)
            if planned:
                data['subject'].append(planned[0]['session']['subject'])
                data['eid'].append(eid)
                data['probe'].append(probe)
                data['planned_x'].append(planned[0]['x'])
                data['planned_y'].append(planned[0]['y'])
                data['planned_depth'].append(planned[0]['depth'])
                data['planned_theta'].append(planned[0]['theta'])
                data['planned_phi'].append(planned[0]['phi'])
                data['micro_x'].append(micro[0]['x'])
                data['micro_y'].append(micro[0]['y'])
                data['micro_depth'].append(micro[0]['depth'])
                data['micro_theta'].append(micro[0]['theta'])
                data['micro_phi'].append(micro[0]['phi'])
                data['hist_x'].append(track_coords.x * 1e6)
                data['hist_y'].append(track_coords.y * 1e6)
                data['hist_depth'].append(track_coords.depth * 1e6)
                data['hist_theta'].append(track_coords.theta)
                data['hist_phi'].append(track_coords.phi)

# Using phi and theta calculate angle in sagittal plane (beta)
x = np.sin(np.array(data['hist_theta']) * np.pi / 180.) * \
    np.sin(np.array(data['hist_phi']) * np.pi / 180.)
y = np.cos(np.array(data['hist_theta']) * np.pi / 180.)
data['hist_beta'] = np.arctan2(x, y) * 180 / np.pi

# Using phi and theta calculate angle in coronal plane (alpha)
x = np.sin(np.array(data['hist_theta']) * np.pi / 180.) * \
    np.cos(np.array(data['hist_phi']) * np.pi / 180.)
y = np.cos(np.array(data['hist_theta']) * np.pi / 180.)
data['hist_alpha'] = np.arctan2(x, y) * 180 / np.pi

data_repeated = pd.DataFrame.from_dict(data)
# Drop these datasets as the histology is way off
data_repeated = data_repeated.drop(np.where(
    data_repeated['subject'] == 'CSHL047')[0]).reset_index(drop=True)
data_repeated = data_repeated.drop(np.where(
    data_repeated['subject'] == 'CSHL054')[0]).reset_index(drop=True)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

# Compute mean deviation in coronal plane
alpha_mean = np.mean(np.abs(data_repeated['planned_theta'] -
                            np.abs(data_repeated['hist_alpha'])))
alpha_std = np.std(np.abs(data_repeated['planned_theta'] -
                          np.abs(data_repeated['hist_alpha'])))

# Compute mean deviation in sagittal plane
beta_mean = np.mean(np.abs(data_repeated['hist_beta']))
beta_std = np.std(np.abs(data_repeated['hist_beta']))

all_ins_entry = np.empty((0, 3))
all_ins_exit = np.empty((0, 3))

# Get the trajectory for the planned repeated site recording
phi_eid = data_repeated['eid'][0]
phi_probe = data_repeated['probe'][0]
phi_traj = one.alyx.rest('trajectories', 'list', session=phi_eid,
                         provenance='Planned', probe=phi_probe)[0]
ins_plan = atlas.Insertion.from_dict(phi_traj)
cax = brain_atlas.plot_tilted_slice(ins_plan.xyz, axis=1, ax=ax1)
cax.plot(ins_plan.xyz[:, 0] * 1e6, ins_plan.xyz[:, 2] * 1e6, 'k', linewidth=4)
sax = brain_atlas.plot_tilted_slice(ins_plan.xyz, axis=0, ax=ax2)
sax.plot(ins_plan.xyz[:, 1] * 1e6, ins_plan.xyz[:, 2] * 1e6, 'k', linewidth=4)
hax = brain_atlas.plot_hslice(ins_plan.xyz[0, 2]-500/1e6, ax=ax3)
hax.plot(ins_plan.xyz[0, 1] * 1e6, ins_plan.xyz[0, 0] * 1e6, color='k',
         marker="o", markersize=6)
hax2 = brain_atlas.plot_hslice(ins_plan.xyz[1, 2], ax=ax4)

# Compute trajectory for each repeated site recording and plot on slice figures
for idx in range(len(data_repeated)):
    phi_eid = data_repeated['eid'][idx]
    phi_probe = data_repeated['probe'][idx]
    phi_subj = data_repeated['subject'][idx]
    phi_traj = one.alyx.rest('trajectories', 'list', session=phi_eid,
                             provenance='Histology track', probe=phi_probe)[0]
    ins = atlas.Insertion.from_dict(phi_traj)

    all_ins_entry = np.vstack([all_ins_entry, ins.xyz[0, :]])
    all_ins_exit = np.vstack([all_ins_exit, ins.xyz[1, :]])
    # channels = bbone.load_channel_locations(phi_eid, one=one,
    # probe=phi_probe)

    # Plot the trajectory for each repeated site recording
    cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, 'y', alpha=0.3)
    sax.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, 'y', alpha=0.3)
    hax.plot(ins.xyz[0, 1] * 1e6, ins.xyz[0, 0] * 1e6, color='y', marker="o",
             markersize=4, alpha=0.6)
    hax2.plot(ins.xyz[1, 1] * 1e6, ins.xyz[1, 0] * 1e6, color='y', marker="o",
              markersize=4, alpha=0.6)

# Compute the mean trajectory across all repeated site recordings
entry_mean = np.mean(all_ins_entry, axis=0)
exit_mean = np.mean(all_ins_exit, axis=0)
ins_mean = np.r_[[entry_mean], [exit_mean]]
# Only consider deviation in ML and AP directions for this analysis
# entry_std = np.std(all_ins_entry, axis=0)
# entry_std[2]=0
# exit_std = np.std(all_ins_exit, axis=0)
# exit_std[2]=0
# ins_upper = np.r_[[entry_mean+entry_std], [exit_mean+exit_std]]
# ins_lower = np.r_[[entry_mean-entry_std], [exit_mean-exit_std]]

# Plot the average track across all repeated site recordings
cax.plot(ins_mean[:, 0] * 1e6, ins_mean[:, 2] * 1e6, 'b', linewidth=4)
sax.plot(ins_mean[:, 1] * 1e6, ins_mean[:, 2] * 1e6, 'b', linewidth=4)
hax.plot(ins_mean[0, 1] * 1e6, ins_mean[0, 0] * 1e6, 'b', marker="o",
         markersize=5)
hax2.plot(ins_mean[1, 1] * 1e6, ins_mean[1, 0] * 1e6, 'b', marker="o",
          markersize=5)
hax2.plot(ins_plan.xyz[1, 1] * 1e6, ins_plan.xyz[1, 0] * 1e6, color='k',
          marker="o", markersize=6)

# Compute targeting error at surface of brain
error_top = all_ins_entry - ins_plan.xyz[0, :]
norm_top = np.sqrt(np.sum(error_top ** 2, axis=1))
top_mean = np.mean(norm_top)*1e6
top_std = np.std(norm_top)*1e6

# Compute targeting error at tip of probe
error_bottom = all_ins_exit - ins_plan.xyz[1, :]
norm_bottom = np.sqrt(np.sum(error_bottom ** 2, axis=1))
bottom_mean = np.mean(norm_bottom)*1e6
bottom_std = np.std(norm_bottom)*1e6

# Add targeting errors to the title of figures
ax1.xaxis.label.set_size(16)
ax1.tick_params(axis='x', labelsize=14)
ax1.yaxis.label.set_size(16)
ax1.tick_params(axis='y', labelsize=14)
ax1.set_title('Targeting error in coronal plane = ' +
              str(np.around(alpha_mean, 1)) + r'$\pm$' +
              str(np.around(alpha_std, 1)) + ' deg', fontsize=18)
ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
ax2.xaxis.label.set_size(16)
ax2.tick_params(axis='x', labelsize=14)
ax2.yaxis.label.set_size(16)
ax2.tick_params(axis='y', labelsize=14)
ax2.set_title('Targeting error in sagittal plane = ' +
              str(np.around(beta_mean, 1)) + r'$\pm$' +
              str(np.around(beta_std, 1)) + ' deg', fontsize=18)
ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
ax3.set_xlabel('ap (um)', fontsize=14)
ax3.tick_params(axis='x', labelsize=14)
ax3.set_ylabel('ml (um)', fontsize=14)
ax3.tick_params(axis='y', labelsize=14)
ax3.set_title('Targeting error at surface = ' +
              str(np.around(top_mean, 1)) + r'$\pm$' +
              str(np.around(top_std, 1)) + ' um', fontsize=18)
ax3.yaxis.set_major_locator(plt.MaxNLocator(4))
ax4.set_xlabel('ap (um)', fontsize=14)
ax4.tick_params(axis='x', labelsize=14)
ax4.set_ylabel('ml (um)', fontsize=14)
ax4.tick_params(axis='y', labelsize=14)
ax4.set_title('Targeting error at probe tip = ' +
              str(np.around(bottom_mean, 1)) + r'$\pm$' +
              str(np.around(bottom_std, 1)) + ' um', fontsize=18)
ax4.yaxis.set_major_locator(plt.MaxNLocator(4))
# ax4.set_ylim((-4000,0))
# ax4.set_xlim((0,-4000))
# ax3.set_ylim((-4000,0))
# ax3.set_xlim((0,-4000))
# ax3.xaxis.ticklabels.set_size(20)
plt.show()
