# fMRI Data Analysis and Visualization
## Introduction
This project focuses on the analysis and visualization of functional magnetic resonance imaging (fMRI) data. By exploring novel methods for visualizing fMRI data, we aim to understand patterns of brain activation and distinguish between different subject groups. The project utilizes two main techniques: gradient maps and Structural Similarity Index (SSIM) analysis.

## Objective
The primary objective of this project is to identify unique areas of brain activation and temporal patterns among different subject groups using fMRI data visualization techniques. We aim to develop methods that preserve both spatial (voxels) and temporal resolution while analyzing the data.

## Challenges
Maintaining spatial and temporal resolution in fMRI data analysis presents a significant challenge. Preserving the integrity of the data while visualizing complex patterns of brain activation requires careful consideration of data processing and visualization techniques.

## Methods
### 1. Gradient Maps
Gradient maps are calculated along the time dimension to detect changes in the fMRI signal over time. The data is divided into smaller matrices to maintain signal significance. Gradient maps are plotted using Matplotlib, allowing for visualization of temporal dynamics in brain activation.

### 2. Structural Similarity Index (SSIM) Analysis
SSIM analysis provides a measure of similarity between two images, commonly used in image processing. In this project, SSIM scores are computed for pairs of small matrices extracted from fMRI data. SSIM scores represent the similarity of brain activation patterns between different subject groups.

## Usage
To replicate the analysis and visualization techniques used in this project, follow these steps:

Ensure you have the necessary Python packages installed, including numpy, scipy, matplotlib, and scikit-image.
Modify the file paths in the provided code to point to your own MATLAB files containing fMRI data.
Run the Python scripts for both gradient maps and SSIM analysis.
Analyze the results to gain insights into brain activation patterns and differences between subject groups.

## Code Details
### Gradient Maps
The grad_map function calculates gradient maps for fMRI data and plots them using Matplotlib.
The data is divided into smaller matrices to preserve signal significance.
Gradient maps visualize temporal dynamics in brain activation patterns.

### SSIM Analysis
The SSIM analysis computes SSIM scores for pairs of small matrices extracted from fMRI data.
The load_mat_files, find_names, and split_to_small_matrices functions prepare the data for SSIM analysis.
The ssim_score function calculates SSIM scores and provides insights into brain activation patterns.

## Conclusion
This project demonstrates the application of gradient maps and SSIM analysis to fMRI data visualization. By exploring novel visualization techniques, we gain insights into brain activation patterns and differences between subject groups. These methods contribute to our understanding of neural processes and have potential applications in neuroscience research.
