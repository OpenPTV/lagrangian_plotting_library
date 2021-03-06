"""
Created on Sat Mar 21 11:06:52 2020

@author: ron
"""


# Imports:

from flowtracks.io import Scene
# from flowtracks.io import save_particles_table

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# ===============================================
#             Loading trajectory lists
# ===============================================

def get_traj_list(file_path):
    '''
    this function loads h5 files and returns a list of trajectories.
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


# ===============================================
#                 Visualizations
# ===============================================
    
        
def plot_3D_quiver(traj_list, v_max, subtract_mean = False, FPS = 500.0,
                   size_factor = 2.0, aspect='equal'):
    '''
    will return a 3D plot of floating quivers that stand for
    the Lagrangin velocity samples.
    
    traj_list - list of trajectories
    v_max - maximum velocity for normalizing colors
    subtract_mean - if true will determine a mean velocity over all samples
                    and remove this from each trajectory. will remove also
                    this mean component of the displacement.
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # estimate arrow lengths:
    i = 0
    while len(traj_list[i]) < 5 and i < len(traj_list):
        i += 1
    L = np.mean(np.linalg.norm(np.gradient(traj_list[i].pos())[0], axis=1))

    if subtract_mean:
        VV = get_mean_velocity(traj_list)
    else:
        VV = np.zeros((3,))

    t0 = traj_list[0].time()[0]
    for tr in traj_list:
        if tr.time()[0] < t0:
            t0 = tr.time()[0]

    cmap = matplotlib.cm.get_cmap('viridis')

    for tr in traj_list:
        tm = tr.time() - t0
        x,y,z = tr.pos()[:,0], tr.pos()[:, 2], tr.pos()[:, 1]
        x,y,z = x - tm*VV[0]/FPS, y - tm*VV[2]/FPS, z - tm*VV[1]/FPS
        u,v,w = tr.velocity()[:, 0], tr.velocity()[:, 2], tr.velocity()[:, 1]
        u, v, w = u - VV[0], v - VV[2], w - VV[1]
        #ax.plot(x,y,z,lw=1,color='k')
        V = 1.0*np.linalg.norm(tr.velocity(), axis=1)/v_max
        V = V * (V <= 1) + (V > 1)
        c = cmap(V)
        ax.quiver(x, y, z, u, v, w, length=L*size_factor,
                  arrow_length_ratio = .5, colors = c)
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    ax.set_zlabel(r'$z$ [m]')
    # ax.set_aspect(aspect)
    return fig, ax


def plot_traj_xy(traj_list,min_len=5, shape='o-', lw=0.5):
    fig, ax = plt.subplots()
    for i in traj_list:
        if len(i) > min_len:
            r = i.pos()
            ax.plot(r[:,0] , r[:,1], shape, ms=1, lw=lw)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    return fig, ax


def plot_traj_xz(traj_list,min_len=5, shape='o-', lw=1):
    fig, ax = plt.subplots()
    for i in traj_list:
        if len(i) > min_len:
            r = i.pos()
            ax.plot(r[:,0] , r[:,2],  shape, ms=1, lw=lw)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$z$ [m]')
    return fig, ax

def plot_traj_yz(traj_list,min_len=5, shape='o-', lw=1):
    fig, ax = plt.subplots()
    for i in traj_list:
        if len(i) > min_len:
            r = i.pos()
            ax.plot(r[:,1] , r[:,2],  shape, ms=1, lw=lw)
    ax.set_xlabel(r'$y$ [m]')
    ax.set_ylabel(r'$z$ [m]')
    ax.set_aspect('equal')
    return fig, ax


# ============================================
#                Statistics
# ============================================


def get_mean_velocity(traj_list):
    '''
    will return a numpy array representing the mean value 
    of velocity components of all trajectories in the list.
    Inputs:
        traj_list - list of trajectories
    Returns:
        mean vel - (vx,vy,vz)
    '''
    # vx,vy,vz = [],[],[]
    
    # for tr in traj_list:
    #     v = tr.velocity()
    #     for i in range(v.shape[0]):
    #         vx.append(v[i,0])
    #         vy.append(v[i,1])
    #         vz.append(v[i,2])
    # V = np.array([np.mean(vx), np.mean(vy), np.mean(vz)])
    # return V

    return np.mean([np.mean(tr.velocity(), axis=0) for tr in traj_list], 
                   axis=0)






def get_vel_p_moment(traj_list, p):
    '''
    will calculate the central p'th moment of velocities
    
    returns -
    mp - < (vx - Vx)^p >, < (vy - Vy)^p >, < (vz - Vz)^p >
    '''

    V = get_mean_velocity(traj_list)
    # vx,vy,vz = [],[],[]

    # for tr in traj_list:
    #     v = tr.velocity() - V
    #     for i in range(v.shape[0]):
    #         vx.append(v[i,0])
    #         vy.append(v[i,1])
    #         vz.append(v[i,2])    
    
    # mp = [np.mean(np.array(vv)**p) for vv in [vx,vy,vz]]
    # return np.array(mp)
    return np.mean([np.mean((tr.velocity()-V)**p, axis=0) for tr in traj_list], 
                axis=0)






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
            vx.append(v[j, 0])
            vy.append(v[j, 1])
            vz.append(v[j, 2])
        if np.amax(np.abs(v)) > M:
            M = np.amax(np.abs(v))
    
    if bin_range==None:
        bin_range=(-M,M)
    
    fig, ax = plt.subplots()
    c = ['b','r','g']
    shp = ['o','d','v']
    lbl = [r'$v_x$',r'$v_y$',r'$v_z$']
    
    for e,i in enumerate([vx, vy, vz]):
        h = np.histogram(i, bins=bins, density=True, range=bin_range)
        x,y = 0.5*(h[1][:-1] + h[1][1:]), h[0]
        m,s = np.mean(i), np.std(i)
        xx = np.arange(-M,M,2.0*M/500)
        ax.plot(x,y,c[e]+shp[e]+'-',lw=0.4,
                label = lbl[e]+r' $\mu = %.3f$ $\sigma = $%0.3f'%(m,s))
        if fit_gaussian:
            ax.plot(xx, gaussian(xx, m, s), c[e], lw = 1.2)
        
    ax.legend()
    ax.set_xlabel(r'$v_i$')
    ax.set_ylabel(r'P($v_i$)')
    
    return fig, ax



# =======================================================
#         Lagrangian 2nd order Structure function
# =======================================================
    
def plot_Dii(traj_list, FPS = 1.0, axis = 0, xlabel=None, ylabel=None):
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
    
    if xlabel==None: ax.set_xlabel(r'$\tau$')
    else: ax.set_xlabel(xlabel)
    
    if ylabel==None: ax.set_ylabel(r'$D_{ii}(\tau)$')
    else: ax.set_ylabel(ylabel)

    return fig, ax, time, D_ii



# ====================================================
#            Lagrangian Autocorrelations
# ====================================================



def plot_velocity_autocorrelation(traj_list, FPS = 1.0, axis = 0,
                                  xlabel=None, ylabel=None):
    '''
    will calculate the Lagrangian autocorrelation function of a
    velocity component ('axis' component), will plot it and finally
    return the results.
    '''
    v_lst = []
    for traj in traj_list:
        v_lst.append(traj.velocity()[:,axis])
    
    rho_ii, N, S = list_corelation(v_lst)
    time = np.arange(len(rho_ii)) / FPS
    
    fig, ax = plt.subplots()
    ax.plot(time, rho_ii)
    
    if xlabel==None: ax.set_xlabel(r'$\tau$')
    else: ax.set_xlabel(xlabel)
    
    if ylabel==None: ax.set_ylabel(r'$\rho_{v,ii}(\tau)$')
    else: ax.set_ylabel(ylabel)
    
    return fig, ax, rho_ii, time







def plot_acceleration_autocorrelation(traj_list, FPS = 1.0, axis = 0,
                                  xlabel=None, ylabel=None):
    '''
    will calculate the Lagrangian autocorrelation function of a
    velocity component ('axis' component), will plot it and finally
    return the results.
    '''
    a_lst = []
    for traj in traj_list:
        a_lst.append(traj.accel()[:,axis])
    
    rho_ii, N, S = list_corelation(a_lst)
    time = np.arange(len(rho_ii)) / FPS
    
    fig, ax = plt.subplots()
    ax.plot(time, rho_ii)
    
    if xlabel==None: ax.set_xlabel(r'$\tau$')
    else: ax.set_xlabel(xlabel)
    
    if ylabel==None: ax.set_ylabel(r'$\rho_{a,ii}(\tau)$')
    else: ax.set_ylabel(ylabel)
    
    return fig, ax, rho_ii, time




# ===================================================
#            General Utilities
# ===================================================


def gaussian(x,m,s):
    return 1.0/np.sqrt(2*np.pi)/s * np.exp(-0.5 * ((x-m)/s)**2)



def average_lists(lsts, get_N = False):
    '''
    retruns an index wise average of
    all the lists in lsts, and the indexed number of averaged values
    '''
    lsts = sorted( lsts, key=len )
    N = len( lsts[-1] )
    indexes = range(N)
    averaged_lsts = []
    N_lsts = [] 
    for i in indexes:
        while len(lsts[0])<i+1: 
            lsts.pop(0)
        temp = 0
        temp_N = 0
        for j in range(len(lsts)):
            temp += lsts[j][i]
            temp_N += 1
        averaged_lsts.append(temp*1.0/len(lsts))
        N_lsts.append(temp_N)
        
    if get_N:
        return averaged_lsts, N_lsts
    else:
        return averaged_lsts
    


def list_corelation(arr_list):
    '''
    returns the array of correlation for a list of arrays as a function of
    time lag:
    
              < (arr(t+x) - <arr(t+x)> )*( arr(t) - <arr(t)> ) >
    R  =  ===============================================================
          sqrt( < (arr(t+x) - <arr(t+x)>)^2 > < (arr(t) - <arr(t)>)^2 > )
          
    ( where <> is average over samples and x is a time (index) lag)
    
    
    returns -
    R - array of correlation coefficients
    S - array of standard deviations for R as a funciton of time
    N - array of number of elements used at each time 
    '''
    N = max( [len(i) for i in arr_list] )
    r = [  [ [],[] ]   for i in range(N)]
    
    for arr in arr_list:
        for val in arr:
            r[0][0].append(val)
            r[0][1].append(val)
        for i in range(1,len(arr)):
            for val in arr[:-i]:
                r[i][0].append(val)
            for val in arr[i:]:
                r[i][1].append(val) 
    R,S,N = [],[],[]
    for i in r:
        if len(i[1]) <= 1:
            R.append(0)
            S.append(0)
            N.append(1)
        else:
            r1 = np.array(i[0]) - np.mean(i[0])
            r2 = np.array(i[1]) - np.mean(i[1])
            R.append( np.mean(r1*r2) / np.sqrt(np.mean(r1**2) * np.mean(r2**2) ) )
            S.append( np.std(r1*r2) / np.sqrt(np.mean(r1**2) * np.mean(r2**2) ) )
            N.append(len(r1))

    return np.array(R), np.array(S), np.array(N)
    
    
    