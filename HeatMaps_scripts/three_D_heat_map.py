import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import pandas as pd
import seaborn as sns

################ Load data and convert mat file ####################
ex_hammetz_dic = scipy.io.loadmat('ExRe_filtered_after_0.1_hammetz.mat')
ex_kosher_dic = scipy.io.loadmat('ExRe_filtered_after_0.1_kosher.mat')
ex_neutral_dic = scipy.io.loadmat('ExRe_filtered_after_0.1_neutral.mat')

rel_hammetz_dic = scipy.io.loadmat('Religious_filtered_after_0.1_hammetz.mat')
rel_kosher_dic = scipy.io.loadmat('Religious_filtered_after_0.1_kosher.mat')
rel_neutral_dic = scipy.io.loadmat('Religious_filtered_after_0.1_neutral.mat')

sec_hammetz_dic = scipy.io.loadmat('Secular_filtered_after_0.1_hammetz.mat')
sec_kosher_dic = scipy.io.loadmat('Secular_filtered_after_0.1_kosher.mat')
sec_neutral_dic = scipy.io.loadmat('Secular_filtered_after_0.1_neutral.mat')

############ From dic, only the data from last key ##############
# Hammetz
ex_hammetz_mat_data = ex_hammetz_dic[list(ex_hammetz_dic.keys())[-1]]  # ndarray of shape=(224770, 144)
rel_hammetz_mat_data = rel_hammetz_dic[list(rel_hammetz_dic.keys())[-1]]
sec_hammetz_mat_data = sec_hammetz_dic[list(sec_hammetz_dic.keys())[-1]]

# Kosher
ex_kosher_mat_data = ex_kosher_dic[list(ex_kosher_dic.keys())[-1]]
rel_kosher_mat_data = rel_kosher_dic[list(rel_kosher_dic.keys())[-1]]
sec_kosher_mat_data = sec_kosher_dic[list(sec_kosher_dic.keys())[-1]]

# Neutral
ex_neutral_mat_data = ex_neutral_dic[list(ex_neutral_dic.keys())[-1]]
rel_neutral_mat_data = rel_neutral_dic[list(rel_neutral_dic.keys())[-1]]
sec_neutral_mat_data = sec_neutral_dic[list(sec_neutral_dic.keys())[-1]]

"""TRs_lengthe_Hammetz = 154
TRs_lengthe_kosher = 144
TRs_lengthe_neutral = 128

Voxels_length_hammetz = 100931
Voxels_length_kosher = 103385
Voxels_length_neutral = 96932"""

######################## Hammetz ###############################
# 3D map for Ex-Religious
# set x,y axis
X_hammetz_ex = np.arange(ex_hammetz_mat_data.shape[1])
Y_hammetz_ex = np.arange(ex_hammetz_mat_data.shape[0])
X_hammetz_ex, Y_hammetz_ex = np.meshgrid(X_hammetz_ex, Y_hammetz_ex)
fig_hammetz_ex, ax_hammetz_ex = plt.subplots(subplot_kw={"projection": "3d"})
surf_hammetz_ex = ax_hammetz_ex.plot_surface(X_hammetz_ex, Y_hammetz_ex, ex_hammetz_mat_data,
                                             cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig_hammetz_ex.colorbar(surf_hammetz_ex, shrink=0.5, aspect=20) # add color bar
plt.title("Hammetz - Ex-Religious")
plt.xlabel("TRs [sec]")
plt.ylabel("Voxels")
plt.show()

# 3D map for Religious
# set x,y axis
X_hammetz_rel = np.arange(ex_hammetz_mat_data.shape[1])
Y_hammetz_rel = np.arange(ex_hammetz_mat_data.shape[0])
X_hammetz_rel, Y_hammetz_rel = np.meshgrid(X_hammetz_rel, Y_hammetz_rel)
fig_hammetz_rel, ax_hammetz_rel = plt.subplots(subplot_kw={"projection": "3d"})
surf_hammetz_rel = ax_hammetz_rel.plot_surface(X_hammetz_rel, Y_hammetz_rel, rel_hammetz_mat_data,
                                               cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig_hammetz_rel.colorbar(surf_hammetz_rel, shrink=0.5, aspect=20) # add color bar
plt.title("Hammetz - Religious")
plt.xlabel("TRs [sec]")
plt.ylabel("Voxels")
plt.show()

# 3D map for Secular
# set x,y axis
X_hammetz_sec = np.arange(ex_hammetz_mat_data.shape[1])
Y_hammetz_sec = np.arange(ex_hammetz_mat_data.shape[0])
X_hammetz_sec, Y_hammetz_sec = np.meshgrid(X_hammetz_sec, Y_hammetz_sec)
fig_hammetz_sec, ax_hammetz_sec = plt.subplots(subplot_kw={"projection": "3d"})
surf_hammetz_sec = ax_hammetz_sec.plot_surface(X_hammetz_sec, Y_hammetz_sec, sec_hammetz_mat_data,
                                               cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig_hammetz_sec.colorbar(surf_hammetz_sec, shrink=0.5, aspect=20) # add color bar
plt.title("Hammetz - Secular")
plt.xlabel("TRs [sec]")
plt.ylabel("Voxels")
plt.show()


######################## Kosher ###############################
# 3D map for Ex-Religious
# set x,y axis
X_kosher_ex = np.arange(ex_kosher_mat_data.shape[1])
Y_kosher_ex = np.arange(ex_kosher_mat_data.shape[0])
X_kosher_ex, Y_kosher_ex = np.meshgrid(X_kosher_ex, Y_kosher_ex)
fig_kosher_ex, ax_kosher_ex = plt.subplots(subplot_kw={"projection": "3d"})
surf_kosher_ex = ax_kosher_ex.plot_surface(X_kosher_ex, Y_kosher_ex, ex_kosher_mat_data,
                                           cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig_kosher_ex.colorbar(surf_kosher_ex, shrink=0.5, aspect=20) # add color bar
plt.title("Kosher - Ex-Religious")
plt.xlabel("TRs [sec]")
plt.ylabel("Voxels")
plt.show()

# 3D map for Religious
# set x,y axis
X_kosher_rel = np.arange(ex_kosher_mat_data.shape[1])
Y_kosher_rel = np.arange(ex_kosher_mat_data.shape[0])
X_kosher_rel, Y_kosher_rel = np.meshgrid(X_kosher_rel, Y_kosher_rel)
fig_kosher_rel, ax_kosher_rel = plt.subplots(subplot_kw={"projection": "3d"})
surf_kosher_rel = ax_kosher_rel.plot_surface(X_kosher_rel, Y_kosher_rel, rel_kosher_mat_data,
                                             cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig_kosher_rel.colorbar(surf_kosher_rel, shrink=0.5, aspect=20) # add color bar
plt.title("Kosher - Religious")
plt.xlabel("TRs [sec]")
plt.ylabel("Voxels")
plt.show()

# 3D map for secular
# set x,y axis
X_kosher_sec = np.arange(ex_kosher_mat_data.shape[1])
Y_kosher_sec = np.arange(ex_kosher_mat_data.shape[0])
X_kosher_sec, Y_kosher_sec = np.meshgrid(X_kosher_sec, Y_kosher_sec)
fig_kosher_sec, ax_kosher_sec = plt.subplots(subplot_kw={"projection": "3d"})
surf_kosher_sec = ax_kosher_sec.plot_surface(X_kosher_sec, Y_kosher_sec, sec_kosher_mat_data,
                                             cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig_kosher_sec.colorbar(surf_kosher_sec, shrink=0.5, aspect=20) # add color bar
plt.title("Kosher - Secular")
plt.xlabel("TRs [sec]")
plt.ylabel("Voxels")
plt.show()


######################## Neutral ###############################
# 3D map for Ex-Religious
# set x,y axis
X_neutral_sec = np.arange(ex_neutral_mat_data.shape[1])
Y_neutral_sec = np.arange(ex_neutral_mat_data.shape[0])
X_neutral_sec, Y_neutral_sec = np.meshgrid(X_neutral_sec, Y_neutral_sec)
fig_neutral_ex, ax_neutral_ex = plt.subplots(subplot_kw={"projection": "3d"})
surf_neutral_ex = ax_neutral_ex.plot_surface(X_neutral_sec, Y_neutral_sec, ex_neutral_mat_data,
                                             cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig_neutral_ex.colorbar(surf_neutral_ex, shrink=0.5, aspect=20) # add color bar
plt.title("Neutral - Ex-Religious")
plt.xlabel("TRs [sec]")
plt.ylabel("Voxels")
plt.show()

# 3D map for Religious
# set x,y axis
X_neutral_rel = np.arange(ex_neutral_mat_data.shape[1])
Y_neutral_rel = np.arange(ex_neutral_mat_data.shape[0])
X_neutral_rel, Y_neutral_rel = np.meshgrid(X_neutral_rel, Y_neutral_rel)
fig_neutral_rel, ax_neutral_rel = plt.subplots(subplot_kw={"projection": "3d"})
surf_neutral_rel = ax_neutral_rel.plot_surface(X_neutral_rel, Y_neutral_rel, rel_neutral_mat_data,
                                               cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig_neutral_rel.colorbar(surf_neutral_rel, shrink=0.5, aspect=20) # add color bar
plt.title("Neutral - Religious")
plt.xlabel("TRs [sec]")
plt.ylabel("Voxels")
plt.show()

# 3D map for secular
# set x,y axis
X_neutral_sec = np.arange(ex_neutral_mat_data.shape[1])
Y_neutral_sec = np.arange(ex_neutral_mat_data.shape[0])
X_neutral_sec, Y_neutral_sec = np.meshgrid(X_neutral_sec, Y_neutral_sec)
fig_neutral_sec, ax_neutral_sec = plt.subplots(subplot_kw={"projection": "3d"})
surf_neutral_sec = ax_neutral_sec.plot_surface(X_neutral_sec, Y_neutral_sec, sec_neutral_mat_data,
                                               cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig_neutral_sec.colorbar(surf_neutral_sec, shrink=0.5, aspect=20) # add color bar
plt.title("Neutral - Secular")
plt.xlabel("TRs [sec]")
plt.ylabel("Voxels")
plt.show()
