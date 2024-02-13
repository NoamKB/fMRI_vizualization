import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def grad_map(matfile):
    # 1. Load 2 files into dic and take the data.
    # 2. Split to small matrices.
    # 3. Find gradients dx, dy for each cell in the small matrices
    # 4. Plot
    dic = scipy.io.loadmat(matfile)
    data = dic[list(dic.keys())[-1]]
    small_matrices  = np.array_split(data, 500, axis=0)
    gradients = [np.gradient(m) for m in small_matrices]  # Gradient of dx, dy for each cell in the small matrices

    for i, (small_matrix, gradient) in enumerate(zip(small_matrices, gradients)):
        fig, ax = plt.subplots(figsize=(25,25))
        #ax.imshow(small_matrix, cmap='hot', interpolation='nearest', aspect='auto')  # Plot the heatmap
        x, y = np.meshgrid(np.arange(small_matrix.shape[1]), np.arange(small_matrix.shape[0]))  # Add arrows to the heatmap
        dx, dy = gradient  # Each gradient is a tuple, representing the grad in x and y direction

        dx = np.where(dx < 0, 0, dx)  # set negative values of the gradient to zero
        #dy = np.where(dy<0, 0, dy)
        dy=np.zeros_like(dy)
        ax.quiver(x, y, dx, dy, scale=30, angles='xy')
        ax.set_title(f"Heatmap for small matrix {i}\n"
                 f"For voxels: {[i * small_matrix.shape[0], (i +1)* small_matrix.shape[1]]}", fontsize=35)

        plt.xlabel('TRs', fontsize=25)
        plt.ylabel('Voxels', fontsize=25)
        plt.show()
        if i == 5:  # out the loop after i heatmaps
            break

file1 = 'ExRe_filtered_after_0.1_hammetz.mat'
grad_map(file1)