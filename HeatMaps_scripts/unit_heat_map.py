import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Load data and convert mat file:
ex_hammetz_dic = scipy.io.loadmat('ExRe_filtered_after_0.1_hammetz.mat')
ex_kosher_dic = scipy.io.loadmat('ExRe_filtered_after_0.1_kosher.mat')
ex_neutral_dic = scipy.io.loadmat('ExRe_filtered_after_0.1_neutral.mat')

rel_hammetz_dic = scipy.io.loadmat('Religious_filtered_after_0.1_hammetz.mat')
rel_kosher_dic = scipy.io.loadmat('Religious_filtered_after_0.1_kosher.mat')
rel_neutral_dic = scipy.io.loadmat('Religious_filtered_after_0.1_neutral.mat')

sec_hammetz_dic = scipy.io.loadmat('Secular_filtered_after_0.1_hammetz.mat')
sec_kosher_dic = scipy.io.loadmat('Secular_filtered_after_0.1_kosher.mat')
sec_neutral_dic = scipy.io.loadmat('Secular_filtered_after_0.1_neutral.mat')


# From dic, only the data from last key.
ex_hammetz_mat_data = ex_hammetz_dic[list(ex_hammetz_dic.keys())[-1]]
rel_hammetz_mat_data = rel_hammetz_dic[list(rel_hammetz_dic.keys())[-1]]
sec_hammetz_mat_data = sec_hammetz_dic[list(sec_hammetz_dic.keys())[-1]]

ex_kosher_mat_data = ex_kosher_dic[list(ex_kosher_dic.keys())[-1]]
rel_kosher_mat_data = rel_kosher_dic[list(rel_kosher_dic.keys())[-1]]
sec_kosher_mat_data = sec_kosher_dic[list(sec_kosher_dic.keys())[-1]]

ex_neutral_mat_data = ex_neutral_dic[list(ex_neutral_dic.keys())[-1]]
rel_neutral_mat_data = rel_neutral_dic[list(rel_neutral_dic.keys())[-1]]
sec_neutral_mat_data = sec_neutral_dic[list(sec_neutral_dic.keys())[-1]]


############################# Heatmap Hammetz ##############################
fig_hammetz, axes_hammetz = plt.subplots(nrows=1, ncols=3,
                                         figsize=(10, 8), sharex=True, sharey=True)  # fig is the window, axes the subplots
# Heat map for ex
plt.subplot(1, 3, 1)
plt.imshow(ex_hammetz_mat_data, interpolation='none', aspect='auto',
           extent=[0, ex_hammetz_mat_data.shape[1], 0, ex_hammetz_mat_data.shape[0]])
plt.colorbar(aspect = 50)

# Heat map for Religious
plt.subplot(1, 3, 2)
plt.imshow(rel_hammetz_mat_data, interpolation='none', aspect='auto',
           extent=[0, rel_hammetz_mat_data.shape[1], 0, rel_hammetz_mat_data.shape[0]])
plt.colorbar(aspect = 50)
# Heat map for Secular
plt.subplot(1, 3, 3)
plt.imshow(sec_hammetz_mat_data, interpolation='none', aspect='auto',
           extent=[0, sec_hammetz_mat_data.shape[1], 0, sec_hammetz_mat_data.shape[0]])
plt.colorbar(aspect = 50)

fig_hammetz.suptitle(f"HeatMap Hammetz - Voxels Vs. TRs"
                     f"\nLeft: Ex-Religious, Middle: Religios, Right: Secular", fontsize=16)

fig_hammetz.supylabel("Voxels", fontsize=14)
fig_hammetz.supxlabel("TRs [sec]", fontsize=14)
fig_hammetz.show()

############################# Heatmap Kosher ##############################
fig_kosher, axes_kosher = plt.subplots(nrows=1, ncols=3,
                                       figsize=(10, 8), sharex=True, sharey=True)  # Create figure and set of subplots

# Heat map for ex
plt.subplot(1, 3, 1)
plt.imshow(ex_kosher_mat_data, interpolation='none', aspect='auto',
           extent=[0, ex_kosher_mat_data.shape[1], 0, ex_kosher_mat_data.shape[0]])
plt.colorbar(aspect = 50)

# Heat map for Religious
plt.subplot(1, 3, 2)
plt.imshow(rel_kosher_mat_data, interpolation='none', aspect='auto',
           extent=[0, rel_kosher_mat_data.shape[1], 0, rel_kosher_mat_data.shape[0]])
plt.colorbar(aspect = 50)

# Heat map for Secular
plt.subplot(1, 3, 3)
plt.imshow(sec_kosher_mat_data, interpolation='none', aspect='auto',
           extent=[0, sec_kosher_mat_data.shape[1], 0, sec_kosher_mat_data.shape[0]])
plt.colorbar(aspect = 50)

fig_kosher.suptitle(f"HeatMap Kosher - Voxels Vs. TRs"
                    f"\nLeft: Ex-religious, Middle: Religious, Right: Secular", fontsize=16)
fig_kosher.supylabel("Voxels", fontsize=14)
fig_kosher.supxlabel("TRs [sec]", fontsize=14)
plt.show()

############################# Heatmap Neutral ##############################
# Create figure and set of subplots
fig_neutral, axes_neutral = plt.subplots(nrows=1, ncols=3,
                                         figsize=(10, 8), sharex=True, sharey=True)
# Heat map for ex
plt.subplot(1, 3, 1)
plt.imshow(ex_neutral_mat_data, interpolation='none', aspect='auto',
           extent=[0, ex_neutral_mat_data.shape[1], 0, ex_neutral_mat_data.shape[0]])
plt.colorbar(aspect = 50)

# Heat map for Religious
plt.subplot(1, 3, 2)
plt.imshow(rel_neutral_mat_data, interpolation='none', aspect='auto',
           extent=[0, rel_neutral_mat_data.shape[1], 0, rel_neutral_mat_data.shape[0]])
plt.colorbar(aspect = 50)

# Heat map for Secular
plt.subplot(1, 3, 3)
plt.imshow(sec_neutral_mat_data, interpolation='none', aspect='auto',
           extent=[0, sec_neutral_mat_data.shape[1], 0, sec_neutral_mat_data.shape[0]])
plt.colorbar(aspect=50)

fig_neutral.suptitle(f"HeatMap Neutral - Voxels Vs. TRs"
                     f"\nLeft: Ex-religious, Middle: Religious, Right: Secular", fontsize=16)
fig_neutral.supylabel("Voxels", fontsize=14)
fig_neutral.supxlabel("TRs [sec]", fontsize=14)
fig_neutral.show()
