from tutorial import get_specific_mice
# %%
mice_m = get_specific_mice.get_mice(sex='m', project_name='ibl_neuropixel_brainwide_01',
                                    training_status='trained')
mice_f = get_specific_mice.get_mice(sex='f', project_name='ibl_neuropixel_brainwide_01',
                                    training_status='trained')
# %%
