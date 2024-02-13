import numpy as np
from skimage.metrics import structural_similarity
import scipy.io
from scipy.io import savemat
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import re


############################## Find the path, load files into dic and take the data in the last key ####################
def load_mat_files(file1, file2, file3, parent_folder, sibling_data_folder):

    # join the file's paths and names
    file_path_Rel = os.path.join(parent_folder, sibling_data_folder, file1)
    file_path_Ex = os.path.join(parent_folder, sibling_data_folder, file2)
    file_path_Sec = os.path.join(parent_folder, sibling_data_folder, file3)

    # load the files into dics and take the data
    dic_rel = scipy.io.loadmat(file_path_Rel)
    data_rel = dic_rel[list(dic_rel.keys())[-1]]
    dic_ex = scipy.io.loadmat(file_path_Ex)
    data_ex = dic_ex[list(dic_ex.keys())[-1]]
    dic_sec = scipy.io.loadmat(file_path_Sec)
    data_sec = dic_sec[list(dic_sec.keys())[-1]]

    return data_rel, data_ex, data_sec


############################## Find groups and video names based on files name ##############################
def find_names(data1, data2, data3):
    name1 = re.search(r'^([a-zA-Z]+)_', data1).group(1)
    name2 = re.search(r'^([a-zA-Z]+)_', data2).group(1)
    name3 = re.search(r'^([a-zA-Z]+)_', data3).group(1)
    video_name = re.search(r'_([a-zA-Z]+)\.', data1).group(1)

    # Create the combined names
    rel_ex = name1 +'_'+ name2
    rel_sec = name1 +'_'+ name3
    ex_sec = name2 +'_'+ name3
    names_and_video_lst = [name1, name2, name3, video_name]
    names_combined_lst = [rel_ex, rel_sec, ex_sec]

    return names_and_video_lst, names_combined_lst


############################# Find the number of voxels and TRs in the large matrix ###################################
def find_data_shape(data):
    total_num_of_voxels = data.shape[0]
    total_num_of_trs = data.shape[1]

    return total_num_of_voxels, total_num_of_trs


############################## Take the big matrix and Split into R*C small matrices ##############################
def split_to_small_matrices(data1, data2, data3, R, C):

    # Split data1 into smaller matrices along the rows and columns
    small_matrices1 = [matrix for matrix in np.array_split(data1, R, axis=0) for matrix in
                      np.array_split(matrix, C, axis=1)]
    small_matrices2 = [matrix for matrix in np.array_split(data2, R, axis=0) for matrix in
                       np.array_split(matrix, C, axis=1)]
    small_matrices3 = [matrix for matrix in np.array_split(data3, R, axis=0) for matrix in
                       np.array_split(matrix, C, axis=1)]


    return small_matrices1, small_matrices2, small_matrices3


############################## Compute the SSIM score for each 2 small matrices ##############################
def ssim_score(small_matrices1, small_matrices2, small_matrices3, R):

    scores_lst_Rel_Ex = []
    scores_lst_Rel_Sec = []
    scores_lst_Ex_Sec = []

    for i in tqdm(range(len(small_matrices1))):
        # Set gaussian_weights = True if you want to give weights for the ssim score.
        # If True, Gaussian kernel with a standard deviation of sigma will be used to weight the scores.
        score_Rel_Ex  = structural_similarity(small_matrices1[i], small_matrices2[i], gaussian_weights = False,
                                           sigma = 1.5, use_sample_covariance = False, channel_axis=True)

        score_Rel_Sec = structural_similarity(small_matrices1[i], small_matrices3[i], gaussian_weights = False,
                                           sigma = 1.5, use_sample_covariance = False, channel_axis=True)

        score_Ex_Sec = structural_similarity(small_matrices2[i], small_matrices3[i], gaussian_weights = False,
                                           sigma = 1.5, use_sample_covariance = False, channel_axis=True)
        scores_lst_Rel_Ex.append(score_Rel_Ex)
        scores_lst_Rel_Sec.append(score_Rel_Sec)
        scores_lst_Ex_Sec.append(score_Ex_Sec)

    list_of_list_of_ssim = [scores_lst_Rel_Ex, scores_lst_Rel_Sec, scores_lst_Ex_Sec]

    df = pd.DataFrame({f'SSIM for {R} voxels: Rel-Ex': scores_lst_Rel_Ex,
                   f'SSIM for {R} voxels: Rel-Sec': scores_lst_Rel_Sec,
                   f'SSIM for {R} voxels: Ex-Sec': scores_lst_Ex_Sec})
    # Send data to excel file
    #file_path = 'C:\Python\Sagol final project\example_mat\SSIM_scores.xlsx'
    #df.to_excel(file_path)
    #print(df)

    return list_of_list_of_ssim


################################## Arange list of SSIM scores in a matrix ###################################
# Every value in mat is the SSIM of the sliced big mat. In size of 1000 voxels and 7 TRs.
def from_ssim_list_to_mat(total_lst_scores, R, C):

    mat_list = []
    for lst in total_lst_scores:
        scores_array = np.array(lst)
        mat_of_ssim = scores_array.reshape(R, C)
        mat_list.append(mat_of_ssim)

    return mat_list


####################### Find min and max ssim score and thier index #######################
def find_min_and_max_SSIM_score(mat_list, total_voxels, total_trs, R, C, names_combined_lst):

    min_lst = []
    max_lst = []
    voxels_range = int(total_voxels / R)
    trs_range = int(total_trs / C)
    for i, (mat, name) in enumerate(zip(mat_list, names_combined_lst)):
        min_value = mat.min()
        max_value = mat.max()
        min_index = mat.argmin()
        max_index = mat.argmax()
        # Calculate the row and column indices of the minimum and maximum values
        min_row, min_col = divmod(min_index, C)
        max_row, max_col = divmod(max_index, C)
        # Calculate the range of voxels and TRs corresponding to the minimum and maximum values
        min_voxels_range = f"{voxels_range * min_row}-{voxels_range * min_row + voxels_range}"
        min_trs_range = f"{trs_range * min_col}-{trs_range * min_col + trs_range}"
        max_voxels_range = f"{voxels_range * max_row}-{voxels_range * max_row + voxels_range}"
        max_trs_range = f"{trs_range * max_col}-{trs_range * max_col + trs_range}"
        # lists the contains the min and max ssim scores respectively
        min_lst.append(min_value)
        max_lst.append(max_value)
        #print(f"For {name} comparison: Min SSIM score={min_value} for voxels: {min_voxels_range} and TRs {min_trs_range}\nMax SSIM score={max_value} for voxels: {max_voxels_range} and TRS {max_trs_range}")

    return min_lst, max_lst


########################################## Plot the SSIM matrix ###############################################
def plot_ssim_mat(mat_list,names_and_video_lst, names_combined_lst, total_voxels, total_trs, R, C, images_folder_path):

    norm = Normalize(vmin=0, vmax=0.5)  # normalize all values to the same range
    voxels_per_cell = round(total_voxels/R)
    trs_per_cell = round(total_trs/C)
    for i, (mat, name) in enumerate(zip(mat_list, names_combined_lst)):
        fig = plt.figure(figsize=(14,10))
        #sns.heatmap(mat, annot=True, annot_kws={"size": 2}, norm=norm, cmap='viridis_r')
        sns.heatmap(mat, norm=norm, cmap='viridis')
        plt.title(f"For {names_and_video_lst[3]} video - SSIM scores for {name}\n"
                  f"Every cell represent {voxels_per_cell} voxels on {trs_per_cell} TRs")
        plt.xlabel("TRs")
        plt.ylabel("Voxels")
        # Save the figure to the save_folder
        #plt.savefig(f"{images_folder_path}/{name}.png")
        plt.show()


############################### Count how many values are higher or smaller the TH ####################################
def count_deviation_from_TH(list_of_list_of_ssim, names_combined_lst, ssim_max_TH, ssim_min_TH):

    # count how many values higher then max TH and lower then min TH
    above_TH_dict = {}
    below_TH_dict = {}
    for i, (lst, name) in enumerate(zip(list_of_list_of_ssim, names_combined_lst)):
        """if name == 'Religious_Secular':
            continue"""
        above_counter = sum(1 for value in lst if value > ssim_max_TH)
        below_counter = sum(1 for value in lst if value < ssim_min_TH)
        above_TH_dict[name] = above_counter
        below_TH_dict[name] = below_counter
    # return a lists of tuples with the count and groups name
    return above_TH_dict, below_TH_dict


############################## Plot Histograms of SSIMN scores distribution #####################################
def plot_PDF_of_SSIM(list_of_list_of_ssim, names_and_video_lst, names_combined_lst, images_folder_path, ssim_max_TH,
                     ssim_min_TH, above_TH_dict, below_TH_dict):

    video_name = names_and_video_lst[3]
    colors = ['r', 'g', 'b']
    # plot hist for each comparison
    f = plt.figure(figsize=(10,10))
    for i, (lst, name) in enumerate(zip(list_of_list_of_ssim, names_combined_lst)):
        # use the if section for presenting only the rel-ex and ex-sec.
        if name == 'Religious_Secular':
            continue
        sns.kdeplot(lst, color=colors[i], label=name)
        #plt.hist(lst, bins=10000, alpha=0.5, color=colors[i], histtype='step', label=name)
        plt.xlim(-0.1, 0.5)
        # Add text showing names and values
        plt.text(0.6, 0.8-0.1*i, f"Above SSIM TH {ssim_max_TH}: {name}: {above_TH_dict[name]}", transform=plt.gca().transAxes)
        plt.text(0.6, 0.77-0.1*i, f"Below SSIM TH {ssim_min_TH}: {name}: {below_TH_dict[name]}", transform=plt.gca().transAxes)
        # Save the figure to the save_folder
        #plt.savefig(f"{images_folder_path}/{name}.png")
    plt.title(f"For {video_name} - SSIM PDF", fontsize=24)
    plt.legend()
    plt.xlabel("SSIM Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.show()
    return f


################################# save all matrices and the avrages to .mat files #####################################
def save_to_matlab_file(mat_list, names_combined_lst, names_and_video_lst):
    matrices_dict = {}
    avrages_dict = {}
    for mat, name in zip(mat_list, names_combined_lst):
        avg_array = np.mean(mat, axis=1)  # avg along the rows
        # Save to dictionary
        matrices_dict[name] = mat
        avrages_dict[name] = avg_array
    # Save to mat files
    savemat(f'matrices_ssim_{names_and_video_lst[3]}.mat', matrices_dict)
    savemat(f'avrages_ssim_{names_and_video_lst[3]}.mat', avrages_dict)
    return matrices_dict, avrages_dict


##############################################################################################################
############################################### Call functions ###############################################

# Define path to the parent and the siblimg folder name to access data
parent_folder = 'C:\Python\Sagol final project\example_mat'
images_folder_path = 'C:\Python\Sagol final project\example_mat\images'
sibling_data_folder = 'FilteredData'
R = 1000  # R controls the number of splited matrices along the rows
C = 22  # R controls the number of splited matrices along the columns

################################################ For Hammetz (H) ################################################
# Files names
file_Rel_H = 'Religious_filtered_after_0.1_hammetz.mat'
file_Ex_H = 'ExRe_filtered_after_0.1_hammetz.mat'
file_Sec_H = 'Secular_filtered_after_0.1_hammetz.mat'

# Load the data (big matrices): takes the files names and the folder names where the data and scripts are.
mat_Rel_H, mat_Ex_H, mat_Sec_H = load_mat_files(file_Rel_H, file_Ex_H, file_Sec_H, parent_folder, sibling_data_folder)

# Find the groups name:
names_and_video_lst_H, names_combined_lst_H = find_names(file_Rel_H, file_Ex_H, file_Sec_H)

# Find how mant voxels (rows) and TRs (columns) are in the large nmatrix
total_voxels_H, total_trs_H = find_data_shape(mat_Rel_H)

# Split to small matrices by factors of R for rows and C for columns, for SSIM score of 1000 voxels and 7 Trs
small_matrix_Rel_H, small_matrix_Ex_H, small_matrix_Sec_H = split_to_small_matrices(mat_Rel_H, mat_Ex_H, mat_Sec_H, R, C)

# Find SSIM score
list_of_list_of_ssim_H = ssim_score(small_matrix_Rel_H, small_matrix_Ex_H, small_matrix_Sec_H, R)

# Create a mat of SSIM scores
ssim_mat_lst_H = from_ssim_list_to_mat(list_of_list_of_ssim_H, R, C)

# find min and max SSIM score
min_score_H, max_score_H = find_min_and_max_SSIM_score(ssim_mat_lst_H, total_voxels_H, total_trs_H, R, C, names_combined_lst_H)

# Plot HeatMap of the ssim matrices
plot_ssim_mat(ssim_mat_lst_H, names_and_video_lst_H, names_combined_lst_H, total_voxels_H, total_trs_H, R, C, images_folder_path)

# Count how many values are higher or smaller the TH
ssim_max_TH_H = 0.3
ssim_min_TH_H = 0
above_TH_dict_H, below_TH_dict_H = count_deviation_from_TH(list_of_list_of_ssim_H, names_combined_lst_H, ssim_max_TH_H, ssim_min_TH_H)

# Plot histogram of SSIM scores
f = plot_PDF_of_SSIM(list_of_list_of_ssim_H, names_and_video_lst_H, names_combined_lst_H, images_folder_path,
                 ssim_max_TH_H, ssim_min_TH_H, above_TH_dict_H, below_TH_dict_H)

# save all matrices to .mat file
matrices_dict_H, avrages_dict_H = save_to_matlab_file(ssim_mat_lst_H, names_combined_lst_H, names_and_video_lst_H)


################################################ For Kosher (K) ################################################
# Files names
file_Rel_K = 'Religious_filtered_after_0.1_kosher.mat'
file_Ex_K = 'ExRe_filtered_after_0.1_kosher.mat'
file_Sec_K = 'Secular_filtered_after_0.1_kosher.mat'

# Load the data (big matrices): takes the files names and the folder names where the data and scripts are.
mat_Rel_K, mat_Ex_K, mat_Sec_K = load_mat_files(file_Rel_K, file_Ex_K, file_Sec_K, parent_folder, sibling_data_folder)

# Find the groups name:
names_and_video_lst_K, names_combined_lst_K = find_names(file_Rel_K, file_Ex_K, file_Sec_K)

# Find how mant voxels (rows) and TRs (columns) are in the large nmatrix
total_voxels_K, total_trs_K = find_data_shape(mat_Rel_K)

# Split to small matrices by factors of R for rows and C for columns, for SSIM score of 100 voxels and 7 Trs
small_matrix_Rel_K, small_matrix_Ex_K, small_matrix_Sec_K = split_to_small_matrices(mat_Rel_K, mat_Ex_K, mat_Sec_K, R, C)

# Find SSIM score
list_of_list_of_ssim_K = ssim_score(small_matrix_Rel_K, small_matrix_Ex_K, small_matrix_Sec_K, R)

# Create a mat of SSIM scores
ssim_mat_lst_K = from_ssim_list_to_mat(list_of_list_of_ssim_K, R, C)

# find min and max SSIM score
min_score_K, max_score_K = find_min_and_max_SSIM_score(ssim_mat_lst_K, total_voxels_K, total_trs_K, R, C, names_combined_lst_K)

# Plot HeatMap of the ssim matrices
plot_ssim_mat(ssim_mat_lst_K, names_and_video_lst_K, names_combined_lst_K, total_voxels_K, total_trs_K, R, C, images_folder_path)

# Count how many values are higher or smaller the TH
ssim_max_TH_K = 0.3
ssim_min_TH_K = 0
above_TH_dict_K, below_TH_dict_K = count_deviation_from_TH(list_of_list_of_ssim_K, names_combined_lst_K, ssim_max_TH_K, ssim_min_TH_K)

# Plot histogram of SSIM scores
plot_PDF_of_SSIM(list_of_list_of_ssim_K, names_and_video_lst_K, names_combined_lst_K, images_folder_path,
                 ssim_max_TH_K, ssim_min_TH_K, above_TH_dict_K, below_TH_dict_K)

# save all matrices and avrages to .mat files
matrices_dict_K, avrages_dict_k = save_to_matlab_file(ssim_mat_lst_K, names_combined_lst_K, names_and_video_lst_K)


################################################ For Neutral (N) ################################################
# Files names
file_Rel_N = 'Religious_filtered_after_0.1_neutral.mat'
file_Ex_N = 'ExRe_filtered_after_0.1_neutral.mat'
file_Sec_N = 'Secular_filtered_after_0.1_neutral.mat'

# Load the data (big matrices): takes the files names and the folder names where the data and scripts are.
mat_Rel_N, mat_Ex_N, mat_Sec_N = load_mat_files(file_Rel_N, file_Ex_N, file_Sec_N, parent_folder, sibling_data_folder)

# Find the groups name:
names_and_video_lst_N, names_combined_lst_N = find_names(file_Rel_N, file_Ex_N, file_Sec_N)

# Find how mant voxels (rows) and TRs (columns) are in the large nmatrix
total_voxels_N, total_trs_N = find_data_shape(mat_Rel_N)
small_matrix_Rel_N, small_matrix_Ex_N, small_matrix_Sec_N = split_to_small_matrices(mat_Rel_N, mat_Ex_N, mat_Sec_N, R, C)

# Find SSIM score
list_of_list_of_ssim_N = ssim_score(small_matrix_Rel_N, small_matrix_Ex_N, small_matrix_Sec_N, R)

# Create a mat of SSIM scores
ssim_mat_lst_N = from_ssim_list_to_mat(list_of_list_of_ssim_N, R, C)

# find min and max SSIM score
min_score_N, max_score_N = find_min_and_max_SSIM_score(ssim_mat_lst_N, total_voxels_N, total_trs_N, R, C, names_combined_lst_N)

# Plot HeatMap of the ssim matrices
plot_ssim_mat(ssim_mat_lst_N, names_and_video_lst_N, names_combined_lst_N, total_voxels_N, total_trs_N, R, C, images_folder_path)

# Count how many values are higher or smaller the TH
ssim_max_TH_N = 0.3
ssim_min_TH_N = 0
above_TH_dict_N, below_TH_dict_N = count_deviation_from_TH(list_of_list_of_ssim_N, names_combined_lst_N, ssim_max_TH_N, ssim_min_TH_N)

# Plot histogram of SSIM scores
plot_PDF_of_SSIM(list_of_list_of_ssim_N, names_and_video_lst_N, names_combined_lst_N, images_folder_path,
                 ssim_max_TH_N, ssim_min_TH_N, above_TH_dict_N, below_TH_dict_N)

# save all matrices and avarges to .mat files
matrices_dict_N, avrages_dict_N = save_to_matlab_file(ssim_mat_lst_N, names_combined_lst_N, names_and_video_lst_N)
