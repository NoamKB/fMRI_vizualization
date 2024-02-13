import scipy.io
import os
import matplotlib.pyplot as plt
import numpy as np

"""# find groups and video names based on files name
def find_names(matfile1, matfile2):
    lst1 = matfile1.split('_')
    lst2 = matfile2.split('_')
    name1 = lst1[0]
    name2 = lst2[0]
    video_name = lst1[4]
    return name1, name2, video_name

# create folder based on files name:
def create_folder(file1, file2):
    path = "C:\Python\Sagol final project\example_mat"
    name1, name2, video_name = find_names(file1, file2)
    folder_name = name1 + '_' + name2 + '_' + video_name +'_images'
    full_path_name = os.path.join(path, folder_name)
    # create a folder
    os.mkdir(full_path_name)
    return folder_name"""

# find gradient map, create plot and save to folder based on files name
# take 2 matfiles as input
def grad_map(matfile1, matfile2):

    # 1. Load 2 files into dic and take the data.
    dic1 = scipy.io.loadmat(matfile1)
    data1 = dic1[list(dic1.keys())[-1]]
    dic2 = scipy.io.loadmat(matfile2)
    data2 = dic2[list(dic2.keys())[-1]]

    # 2. Split to small matrices.
    small_matrices1  = np.array_split(data1, 200, axis=0)
    small_matrices2  = np.array_split(data2, 200, axis=0)

    # 3. Find gradients dx, dy for each cell in the small matrices
    gradients1 = [np.gradient(m) for m in small_matrices1]
    gradients2 = [np.gradient(m) for m in small_matrices2]

    # call the find_groups_name function to get the files name:
    name1, name2, video_name = find_names(matfile1, matfile2)

    # create folder
    folder_name = create_folder(matfile1, matfile2)

    for i, (small_matrix1, gradient1, small_matrix2, gradient2) in \
            enumerate(zip(small_matrices1, gradients1, small_matrices2, gradients2)):
        # voxels range of each small matrix
        voxels_range = [i * small_matrix1.shape[0], (i + 1) * small_matrix1.shape[0]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,15))
        # Plot subplot 1
        #extent1 = [0, small_matrix1.shape[1], (i + 1) * small_matrix1.shape[0], i * small_matrix1.shape[0]]
        #ax1.imshow(small_matrix1, cmap='hot', interpolation='nearest', aspect='auto', extent=extent1)  # For the heatmap
        x1, y1 = np.meshgrid(np.arange(small_matrix1.shape[1]), np.arange(small_matrix1.shape[0]))  # Add arrows
        dx1, dy1 = gradient1  # Each gradient is a tuple, representing the grad in x and y direction

        dx1 = np.where(dx1 < 0, 0, dx1)  # set negative values of the gradient to zero
        dy1 = np.zeros_like(dy1)
        ax1.quiver(x1[::4, ::2], y1[::4, ::2], dx1[::4, ::2], dy1[::4, ::2], scale=20, angles='xy', headlength=20, headwidth=15)
        #ax1.quiver(x1, y1, dx1, dy1, scale=30, angles='xy')
        ax1.set_title(f"For Group {name1} - Gradient Map for small matrix {i}\n"
                 f"For voxels: {voxels_range}", fontsize=30)
        ax1.invert_yaxis()
        ax1.set_xlabel('TRs', fontsize=25)
        ax1.set_ylabel('Voxels', fontsize=25)

        # Plot subplot 2
        #extent2 = [0, small_matrix2.shape[1], (i + 1) * small_matrix2.shape[0], i * small_matrix2.shape[0]]
        #ax2.imshow(small_matrix2, cmap='hot', interpolation='nearest', aspect='auto', extent=extent2)  # For the heatmap
        x2, y2 = np.meshgrid(np.arange(small_matrix2.shape[1]), np.arange(small_matrix2.shape[0]))  # Add arrows
        dx2, dy2 = gradient2  # Each gradient is a tuple, representing the grad

        dx2 = np.where(dx2 < 0, 0, dx2)  # set negative values of the gradient to zero
        dy2 = np.zeros_like(dy2)
        ax2.quiver(x2[::4, ::2], y2[::4, ::2], dx2[::4, ::2], dy2[::4, ::2], scale=20, angles='xy', headlength=20, headwidth=15)
        # for display with no jumps
        #ax2.quiver(x2, y2, dx2, dy2, scale=30, angles='xy')
        ax2.set_title(f"For Group {name2}: Gradient Map for small matrix {i}\n"
                      f"For voxels: {voxels_range}", fontsize=30)
        ax2.invert_yaxis()
        ax2.set_xlabel('TRs', fontsize=25)
        ax2.set_ylabel('Voxels', fontsize=25)

        plt.show()


# define files
file1 = 'ExRe_filtered_after_0.1_hammetz.mat'
file2 = 'Religious_filtered_after_0.1_hammetz.mat'
file3 = 'Secular_filtered_after_0.1_hammetz.mat'
# call function
grad_map(file1, file2)
grad_map(file2, file3)
grad_map(file1, file3)