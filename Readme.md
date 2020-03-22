# Lagrangian plotting utilities 

Developed by Ron Shnapp based on `flowtracks` package http://github.com/openptv/postptv
The package uses python to generate graphs for a Lagrangian analysis of turbulent flows.


## Installation instructions

If you use `conda`:  

        conda create -n lagrangian_plotting python=3.7
        conda activate lagrangian_plotting
        conda install numpy scipy matplotlib jupyter jupyterlab nb_conda h5py tables
        pip install git+https://github.com/openptv/postptv.git
        jupyter notebook Tutorials.ipynb





## Contains:

1. A python script with various functions used to manage Lagrangian data, and to generate the plots
2. An example set of 200 Trajectories. The trajectories were obtained from the Johns Hopkins Turbulence Database (JHTDB, HIT DNS), see Shnapp & Liberzon 2018, PRL.
3. An ipython notebook with a tutorial showing how to use the package.
