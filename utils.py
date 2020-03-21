#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:06:52 2020

@author: ron
"""


# Imports:

from flowtracks.io import Scene
from flowtracks.io import save_particles_table
import numpy as np
import matplotlib.pyplot as plt



# Loading trajectory lists:

def get_traj_list(file_path):
    '''
    this function loads h5 files and returns a list of trajetories.
    file_path is either the path to a single h5 file, or a list of links
    to multiple h5 files.
    '''
    if type(file_path) == str:
        file_path = [file_path]
    t = []
    for i in file_path:
        s = Scene(i)
        for traj in s.iter_trajectories():
            t.append(traj)
    return t





# ============================================
#                Statistics
# ============================================



def get_mean_velocity(traj_list):
    '''
    will return a numpy array representing the mean value of velocityies 
    for all trajectories in a list.
    
    returns -
    mean vel - (vx,vy,vz)
    '''
    vx,vy,vz = [],[],[]
    
    for tr in traj_list:
        v = tr.velocity()
        for i in range(v.shape[0]):
            vx.append(v[i,0])
            vy.append(v[i,1])
            vz.append(v[i,2])
    V = np.array([np.mean(vx), np.mean(vy), np.mean(vz)])
    return V






def get_vel_p_moment(traj_list, p):
    '''
    will calculate the central p'th moment of velocities
    
    returns -
    mp - < (vx - Vx)^p >, < (vy - Vy)^p >, < (vz - Vz)^p >
    '''

    V = get_mean_velocity(traj_list)
    vx,vy,vz = [],[],[]

    for tr in traj_list:
        v = tr.velocity() - V
        for i in range(v.shape[0]):
            vx.append(v[i,0])
            vy.append(v[i,1])
            vz.append(v[i,2])    
    
    mp = [np.mean(np.array(vv)**p) for vv in [vx,vy,vz]]
    return np.array(mp)





def plot_vel_pdfs(traj_list, fit_gaussian=True, bins=100, bin_range=None):
    '''
    will generate a pdf of trajectory vecolicties and if specified 
    by (fit_gaussian = True) will fit a gaussian to the data
    '''
    vx,vy,vz = [],[],[]
    M = -1.0
    for i in traj_list:
        v = i.velocity()
        for j in range(v.shape[0]):
            vx.append(v[j,0])
            vy.append(v[j,1])
            vz.append(v[j,2])
        if np.amax(np.abs(v)) > M:
            M = np.amax(np.abs(v))
    
    if bin_range==None:
        bin_range=(-M,M)
    
    fig, ax = plt.subplots()
    c = ['b','r','g']
    shp = ['o','d','v']
    lbl = [r'$v_x$',r'$v_y$',r'$v_z$']
    
    for e,i in enumerate([vx,vy,vz]):
        h = np.histogram(i,bins=bins, normed = True, range=bin_range)
        x,y = 0.5*(h[1][:-1] + h[1][1:]), h[0]
        m,s = np.mean(i), np.std(i)
        xx = np.arange(-M,M,2.0*M/500)
        ax.plot(x,y,c[e]+shp[e]+'-',lw=0.4,
                label=lbl[e]+r' $\mu = %.3f$ $\sigma = $%0.3f'%(m,s))
        if fit_gaussian:
            ax.plot(xx, gaussian(xx, m, s), c[e], lw = 1.2)
        
    ax.legend()
    ax.set_xlabel(r'$v_i$')
    ax.set_ylabel(r'P($v_i$)')
    
    return fig, ax


def gaussian(x,m,s):
    return 1.0/np.sqrt(2*np.pi)/s * np.exp(-0.5 * ((x-m)/s)**2)




# =======================================================
#         Lagrangian 2nd order Structure function
# =======================================================
    
def plot_Dii(traj_list, FPS = 500.0, axis = 0):
    '''
    will plot and return the matplotlib axis object for the 2nd order
    Lagrangian structures function:
        D_ii = < (v_i(t + x) - v_i(t))^2 >
    here i is the axis for the function calculation 
    '''
    D = []
    for i in traj_list:
        v = i.velocity()[:,axis]
        D.append([0])
        for j in range(len(i) - 1):
            D[-1].append( np.mean( (v[:-(j+1)] - v[(j+1):])**2 ) )
    D_ii = average_lists(D)
    time = np.arange(len(D_ii))/FPS
                    
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(D_ii))/FPS, D_ii,'-o')
    ax.set_xlabel(r'$\tau$ [s]')
    ax.set_ylabel(r'$D_{xx}(\tau) = \langle ( v_x(t+\tau) - v_x(t) )^2 \rangle$')
    return fig, ax, time, D_ii




