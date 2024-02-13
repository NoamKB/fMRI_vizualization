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
ex_hammetz_mat_data = ex_hammetz_dic[list(ex_hammetz_dic.keys())[-1]]  # ndarray of shape=(224770, 144)
rel_hammetz_mat_data = rel_hammetz_dic[list(rel_hammetz_dic.keys())[-1]]
sec_hammetz_mat_data = sec_hammetz_dic[list(sec_hammetz_dic.keys())[-1]]

ex_kosher_mat_data = ex_kosher_dic[list(ex_kosher_dic.keys())[-1]]
rel_kosher_mat_data = rel_kosher_dic[list(rel_kosher_dic.keys())[-1]]
sec_kosher_mat_data = sec_kosher_dic[list(sec_kosher_dic.keys())[-1]]

ex_neutral_mat_data = ex_neutral_dic[list(ex_neutral_dic.keys())[-1]]
rel_neutral_mat_data = rel_neutral_dic[list(rel_neutral_dic.keys())[-1]]
sec_neutral_mat_data = sec_neutral_dic[list(sec_neutral_dic.keys())[-1]]

# Matrix shapes - based on data of the same shape.
voxels_length = ex_hammetz_mat_data.shape[0] # 100931
TRs_length = ex_hammetz_mat_data.shape[1] # 154

# Graphical presentation
voxel_step = 50000
count = 100
# All TRs, jump in 50,000 voxels every iteration
for voxel in range(voxels_length)[:200000:voxel_step]:

    # Heat map for Secular
    plt.imshow(sec_mat_data[voxel:voxel+voxel_step,:], interpolation = 'none', aspect='auto', extent=[0,TRs_length,voxel,voxel+voxel_step])
    plt.title(f"Secular BOLD signal [PDU]- 144 TRs Vs. {voxel}-{voxel+voxel_step} Voxels")
    plt.ylabel("Voxels")
    plt.xlabel("TRs")
    plt.colorbar(orientation ='horizontal' ,pad = 0.15, aspect = 20)
    plt.show()

    # Heat map for Religious
    plt.imshow(rel_mat_data[voxel:voxel+voxel_step,:], interpolation = 'none', aspect='auto', extent=[0,TRs_length,voxel,voxel+voxel_step])
    plt.title(f"Religious BOLD signal [PDU]- 144 TRs Vs. {voxel}-{voxel+voxel_step} Voxels")
    plt.ylabel("Voxels")
    plt.xlabel("TRs")
    plt.colorbar(orientation ='horizontal' ,pad = 0.15, aspect = 20)
    plt.show()

    # Heat map for Religious - Secular
    #plt.subplot(count+2)
    plt.imshow(data_diff_mat[voxel:voxel+voxel_step,:], interpolation = 'none', aspect='auto', extent=[0,TRs_length,voxel,voxel+voxel_step])
    plt.title(f"Difference Rel-Sec BOLD signal [PDU]- 144 TRs Vs. {voxel}-{voxel+voxel_step} Voxels")
    plt.ylabel("Voxels")
    plt.xlabel("TRs")
    plt.colorbar(orientation ='horizontal' ,pad = 0.15, aspect = 20)
    plt.show()
    count += 3

# Difference - All voxels, jump in 2 TRs every iteration:
tr_step = 5
for tr in range(TRs_length)[::tr_step]:
    plt.imshow(data_diff_mat[:,tr:tr+tr_step], interpolation = 'none', aspect='auto', extent=[tr,tr+tr_step,0,voxels_length])
    plt.title(f"Difference Rel-Sec BOLD signal [PDU]- {tr}-{tr+tr_step} TRs Vs. all Voxels")
    plt.ylabel("Voxels")
    plt.xlabel("TRs")
    ax = plt.gca()
    plt.colorbar(orientation ='horizontal' ,pad = 0.5, aspect = 20)
    plt.show()

