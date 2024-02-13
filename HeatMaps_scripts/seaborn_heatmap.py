import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sys

# Load data and convert mat file:
rel_dic = scipy.io.loadmat('Religious_mat.mat')  #return dictionary with all the headers as keys
sec_dic = scipy.io.loadmat('Secular_mat.mat') #return dictionary with all the headers as keys
exrel_dic = scipy.io.loadmat('ExRe_mat.mat')

#Take only the data associated with religious or secular key
rel_mat_data = rel_dic['Religious']  # ndarray of shape=(224770, 144)
sec_mat_data = sec_dic['Secular']  # ndarray of shape=(224770, 144)
ex_mat_data = exrel_dic['ExRe']

# Matrix shape:
voxels_length = rel_mat_data.shape[0] # 224770
TRs_length = rel_mat_data.shape[1] # 144

# Difference matrix:
diff_rel_sec_data_mat = np.subtract(rel_mat_data, sec_mat_data)
diff_rel_ex_data_mat = np.subtract(rel_mat_data, ex_mat_data)
diff_ex_sec_data_mat = np.subtract(ex_mat_data, sec_mat_data)

# Create Data frame for heatmap
sec_data_frame = pd.DataFrame(sec_mat_data)
rel_data_frame = pd.DataFrame(rel_mat_data)
ex_data_frame = pd.DataFrame(ex_mat_data)

diff_rel_sec_data_frame = pd.DataFrame(diff_rel_sec_data_mat)
diff_rel_ex_data_frame = pd.DataFrame(diff_rel_ex_data_mat)
diff_ex_sec_data_frame = pd.DataFrame(diff_ex_sec_data_mat)

################################# PLOT HEATMAP ##############################

fig , axes = plt.subplots(1,3, figsize=(10,15) ,sharex = True, sharey=True)
# For Sec
plt.subplot(1,3,1)
sns.set(font_scale=1.5)
sns.heatmap(sec_data_frame, cmap='Blues' ,cbar=False)
plt.title("Sec", fontsize=16)

# For Rel
plt.subplot(1,3,2)
sns.set(font_scale=1.5)
sns.heatmap(rel_data_frame, cmap='Blues', cbar=False)
plt.title("Rel", fontsize=16)

#for exrel
plt.subplot(1,3,3)
sns.set(font_scale=1.5)
sns.heatmap(ex_data_frame, cmap='Blues')
plt.title("Ex", fontsize=16)

fig.supylabel("Voxels", fontsize=16)
fig.supxlabel("TRs", fontsize=16)
fig.show()

################################### FOR DIFF #######################################
fig_diff , axes_diff = plt.subplots(1,3, figsize=(10,15) ,sharex = True, sharey=True)
# Rel - Sec
plt.subplot(1,3,1)
sns.set(font_scale=1.5)
sns.heatmap(diff_rel_sec_data_frame, cmap='Blues' ,cbar=False)
plt.title("Diff Rel - Sec", fontsize=16)

# Rel - Ex
plt.subplot(1,3,2)
sns.set(font_scale=1.5)
sns.heatmap(diff_rel_ex_data_frame, cmap='Blues' ,cbar=False)
plt.title("Diff Rel - Ex", fontsize=16)

# ex - Sec
plt.subplot(1,3,3)
sns.set(font_scale=1.5)
sns.heatmap(diff_ex_sec_data_frame, cmap='Blues')
plt.title("Diff Ex - Sec", fontsize=16)

fig_diff.supylabel("Voxels", fontsize=16)
fig_diff.supxlabel("TRs", fontsize=16)
fig_diff.show()

a = np.array([[1,2,2],[1,1,1]])