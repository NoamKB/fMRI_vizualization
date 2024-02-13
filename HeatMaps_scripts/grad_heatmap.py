import scipy.io
import matplotlib.pyplot as plt
import numpy as np


ex_hammetz_dic = scipy.io.loadmat('ExRe_filtered_after_0.1_hammetz.mat')
ex_hammetz_mat_data = ex_hammetz_dic[list(ex_hammetz_dic.keys())[-1]]

small_matrices  = np.array_split(ex_hammetz_mat_data, 500, axis=0)

gradients = [np.gradient(m) for m in small_matrices]

for i, (small_matrix, gradient) in enumerate(zip(small_matrices, gradients)):
    fig, ax = plt.subplots(figsize=(25, 25))

    # Add arrows to the heatmap
    x, y = np.meshgrid(np.arange(small_matrix.shape[1]), np.arange(small_matrix.shape[0]))
    dx, dy = gradient  # Each gradient is a tuple, representing the grad in x and y direction
    dx = np.where(dx < 0, 0, dx)  # set negative values of the gradient to zero
    dy = np.where(dy, 0, dy)  # sey y grad to 0

    ax.imshow(dx, cmap='hot', interpolation='nearest', aspect='auto')
    #ax.quiver(x, y, dx, dy, scale=20, angles='xy', )

    ax.set_title(f"Heatmap for small matrix {i}\n"
                 f"For voxels: {[i * small_matrix.shape[0], (i + 1) * small_matrix.shape[1]]}", fontsize=35)

    plt.xlabel('TRs', fontsize=25)
    plt.ylabel('Voxels', fontsize=25)
    plt.show()
