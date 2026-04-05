# Oprimization of the Signal Analysis Pipeline for the observation of Hydrogen

**The following project is part of the collective effort of the members of the BEAM-Astro team**

## Installation list:

- CUDA toolkit 13.1  
- Visual Studio Code
- Python 3.1
- Jupyter Notebooks for Visual Studio Code

## Notes on the selected parallelization technology

The CUDA framework was chosen because it provides an easier to understand, more robust and stable way to parallelize existing code and improve performance. There are also a lot of available resources to help with the implementation of the necessary algorithms and operations. 

## Techniques used for optimization of the kernel

i. Shared memory for more efficient utillization of each *block*

ii. Grid Stride loop to prevent overflow of the data 
