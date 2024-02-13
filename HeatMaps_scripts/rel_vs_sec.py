import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# Load data and convert mat file:
rel_dic = scipy.io.loadmat('FilteredData/Religious_filtered_after_0.1_hammetz.mat')  # Return dictionary with all the headers as keys
sec_dic = scipy.io.loadmat('FilteredData/Secular_filtered_after_0.1_hammetz.mat')
ex_dic = scipy.io.loadmat('FilteredData/ExRe_filtered_after_0.1_hammetz.mat')

#Take only the data associated with religious or secular key
rel_mat_data = rel_dic['Religious_filtered']  # ndarray of shape=(100931, 154)
sec_mat_data = sec_dic['Secular_filtered']

# Matrix shape:
voxels_length = rel_mat_data.shape[0] # 224770
TRs_length = rel_mat_data.shape[1] # 144

"""# For Religious -  line plot of voxels for specific range of TRs (all  columns)
print("===============================Religious==================================")
for tr in range(100,103):
    averag_rel = np.average(rel_mat_data[200:300, tr])
    std_rel = np.std(rel_mat_data[200:300, tr])
    print(f"For Voxels 200-300 - Average BOLD signal, of the {tr} TR, Religious: {averag_rel}")
    #print(f"For Voxels 200-300 - Std of BOLD signal, of the {tr} TR, Religious: {std_rel}")
    plt.plot( range(voxels_length)[200:300] , rel_mat_data[200:300, tr])
    plt.title(f"Religious - 100 voxels for the {tr-2}-{tr+1} TRs")
    plt.xlabel("Voxels number")
    plt.ylabel("BOLD value - PDU unit")
plt.show()
print("===============================Secular==================================")
# For Seculr -  line plot of voxels for specific range of TRs (all  columns)
for tr in range(100,103):
    averag_sec = np.average(sec_mat_data[200:300, tr])
    std_sec = np.std(sec_mat_data[200:300, tr])
    print(f"For Voxels 200-300 - Average BOLD signal, of the {tr} TR, Seculr: {averag_sec}")
    #print(f"For Voxels 200-300 - Std of BOLD signal, of the {tr} TR, Seculr: {std_sec}")
    plt.plot( range(voxels_length)[200:300] , sec_mat_data[200:300, tr])
    plt.title(f"Secular - 100 voxels for the {tr-2}-{tr+1} TRs")
    plt.xlabel("Voxels number")
    plt.ylabel("BOLD value - PDU unit")
plt.show()
print("===============================Differnece Rel-Sec==================================")
# For difference Rel - Sec -  line plot of voxels for specific range of TRs (all  columns)
for tr in range(100,103):
    averag_sec = np.average(data_diff_mat[200:300, tr])
    std_sec = np.std(data_diff_mat[200:300, tr])
    print(f"For Voxels 200-300 - Average BOLD signal, of the {tr} TR, Difference Rel-Sec: {averag_sec}")
    #print(f"For Voxels 200-300 - Std of BOLD signal, of the {tr} TR, Difference Rel-Sec: {std_sec}")
    plt.plot( range(voxels_length)[200:300] , data_diff_mat[200:300, tr])
    plt.title(f"Difference Rel-Sec - 100 voxels for the {tr-2}-{tr+1} TRs")
    plt.xlabel("Voxels number")
    plt.ylabel("BOLD value - PDU unit")
plt.show()"""


# For 144 TRs - the BOLD signal of the 100-103 voxels:
""""# For Religious
print("===============================Religious==================================")
for voxel in range(100, 103):
    averag_rel = np.average(rel_mat_data[voxel, range(TRs_length)])  # avg of BOLD in the voxel index along all the TRs
    std_rel = np.std(rel_mat_data[voxel, range(TRs_length)])
    print(f"For 144 TRs - Average BOLD signal, of the {voxel} voxel, Religious: {averag_rel}")
    #print(f"For 144 TRs - Std of BOLD signal, of the {voxel} voxel, Religious: {std_rel}")
    plt.plot(range(TRs_length), rel_mat_data[range(TRs_length),voxel])
    plt.title(f"Religious - 144 BOLD values for Trs for the {voxel - 2}-{voxel + 1} voxel")
    plt.ylabel("BOLD for Voxels - PDU unit")
    plt.xlabel("TRs number")
plt.show()

print("===============================Secular==================================")
# For Seculr
for voxel in range(100, 103):
    averag_sec = np.average(sec_mat_data[voxel, range(TRs_length)])
    std_sec = np.std(sec_mat_data[voxel, range(TRs_length)])
    print(f"For 144 TRs - Average BOLD signal, of the {voxel} voxel, Seculr: {averag_sec}")
    #print(f"For 144 TRs - Std of BOLD signal, of the {voxel} voxel, Seculr: {std_sec}")
    plt.plot(range(TRs_length), sec_mat_data[range(TRs_length),voxel])
    plt.title(f"Secular - 144 BOLD value for Tr for the {voxel - 2}-{voxel + 1} voxel")
    plt.ylabel("BOLD for Voxels - PDU unit")
    plt.xlabel("TRs Number")
plt.show()
print("===============================Differnece Rel-Sec==================================")
# For difference Rel - Sec
for voxel in range(100, 103):
    averag_sec = np.average(data_diff_mat[voxel, range(TRs_length)])
    std_sec = np.std(data_diff_mat[voxel, range(TRs_length)])
    print(f"For 144 TRs- Average BOLD signal, of the {voxel} TR, Difference Rel-Sec: {averag_sec}")
    #print(f"For 144 TRs - Std of BOLD signal, of the {voxel} TR, Difference Rel-Sec: {std_sec}")
    plt.plot(range(TRs_length), data_diff_mat[range(TRs_length),voxel])
    plt.title(f"Difference Rel-Sec - 144 BOLD values for Tr for the {voxel - 2}-{voxel + 1} voxel")
    plt.ylabel("BOLD for Voxels - PDU unit")
    plt.xlabel("TRs Number")
plt.show()"""


# Scatter plot
"""for tr in range(TRs_length - 1):  # loop over 144 TRs.
    plt.scatter(sec_mat_data[200000:201000,tr], rel_mat_data[200000:201000,tr])
    plt.title(f"Secular Vs. Religious -(1000-2000) of the {tr} TR")
    plt.xlabel(f"200k-201k Voxels of {tr} TR - Secular group")
    plt.ylabel(f"200k-201k Voxels of {tr} TR - Religious group")
# Display the plot
    plt.show()"""

#voxel_range = sec_mat_data[1000:2000:100,:]

# Graphically presentation
voxel_step = 50000
count = 100
for voxel in range(voxels_length)[:200000:voxel_step]:
    # Creat color image for secular
    #plt.subplot(count)
    plt.imshow(sec_mat_data[voxel:voxel+voxel_step,:], interpolation = 'none', aspect='auto')
    plt.title(f"Secular BOLD signal [PDU]- 144 TRs Vs. {voxel}-{voxel+voxel_step} Voxels")
    plt.ylabel("Voxels")
    plt.xlabel("TRs")
    plt.colorbar(orientation ='horizontal' ,pad = 0.15)
    plt.show()

    # Create histogram for secular
    """plt.hist(sec_mat_data[voxel:voxel+voxel_step,:], bins=10)
    plt.title(f"Secular Histogram of all Trs in range of {voxel}-{voxel+voxel_step} voxels")
    plt.xlabel("BOLD value - PDU unit")
    plt.ylabel("Number of appearnces")
    plt.show()"""

    # Creat color image for religious
    #plt.subplot(count+1)
    plt.imshow(rel_mat_data[voxel:voxel+voxel_step,:], interpolation = 'none', aspect='auto')
    plt.title(f"Religious BOLD signal [PDU]- 144 TRs Vs. {voxel}-{voxel+voxel_step} Voxels")
    plt.ylabel("Voxels")
    plt.xlabel("TRs")
    plt.colorbar(orientation ='horizontal' ,pad = 0.15)
    plt.show()

    # Create histogram for religious
    """plt.hist(rel_mat_data[voxel:voxel+voxel_step,:], bins=10)
    plt.title(f"Religious Hisrogram for all Trs in range of {voxel}-{voxel+voxel_step} voxels")
    plt.xlabel("BOLD value - PDU unit")
    plt.ylabel("Number of appearnces")
    plt.show()"""

    # Creat color image for religious
    #plt.subplot(count+2)
    plt.imshow(data_diff_mat[voxel:voxel+voxel_step,:], interpolation = 'none', aspect='auto')
    plt.title(f"Difference Rel-Sec BOLD signal [PDU]- 144 TRs Vs. {voxel}-{voxel+voxel_step} Voxels")
    plt.ylabel("Voxels")
    plt.xlabel("TRs")
    plt.colorbar(orientation ='horizontal' ,pad = 0.15)
    plt.show()
    count += 3