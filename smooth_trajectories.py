#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 21:47:25 2018

@author: ron


implementation of 2 methods for smoothing trajectories.

1) with a polynomial fitting, after Luti et al 2005
2) with savizky golay

"""

import numpy as np
from flowtracks.io import Trajectory
from scipy.signal import savgol_filter as sgf


def smooth_traj_poly(traj, window, polyorder, FPS, cut_edges = True):
    '''
    will smooth a flowtracks trajectory.
    smooth the position using a mooving polynomial, where
    the velocities and accelerations are calculated by
    differentiating the polynomial coeficcients analytically.
    
    the new trajectory is shorter by one window size

    the smoothed trajectory has also a new property - residuals.
    this is the original position minus the smoothed position of the
    particle and can be used for error assessments.
    
    returns-
    new_traj - a smoothed trajectory object
    '''
    
    if type(window) == int:
        window = (window, window, window)
    
    if type(polyorder) == int:
        polyorder = (polyorder, polyorder, polyorder)
    
    if type(window) != tuple: 
        raise TypeError('window must be either integer or tuple')
    
    if type(polyorder) != tuple: 
        raise TypeError('polyorder must be either a number or tuple')
    
    test = [w%2 != 1 for w in window]
    if sum(test) != 0:
        raise ValueError('each window must be a positive odd integer.')
    
    test = [polyorder[i] > window[i] for i in [0,1,2]]
    if sum(test) != 0:
        raise ValueError('polyorder cannot be larger than window.')
        
    N = len(traj)
    test = [w > N for w in window]
    if sum(test) != 0:
        raise ValueError('window cannot be larger than the trajectory length.')
    
    
    
    W = max(window)
    p = traj.pos()
    if cut_edges: new_N = N - W + 1
    elif cut_edges==False : new_N = N 
    new_pos, new_vel, new_acc = np.empty((new_N ,3)), np.empty((new_N,3)), np.empty((new_N,3))
    
    sequence = zip(range(3), window, polyorder)
    
    
    # smoothing the trajectory with cutting the edges of the trajectory
    if cut_edges:
        for e,win,po in sequence:
            time_ = np.arange(win)
            ev_point = float(win/2)**np.arange(po + 1)[::-1]
            skip = (W-win)/2
            
            Deriv_mat = []
            for i in range(po+1):
                a = [0.0 for j in range(po+1)]
                if i!=0:
                    a[i-1] = po - (i-1)
                Deriv_mat.append(a)
            Deriv_mat = np.array(Deriv_mat)
            
            for i in range( new_N ):
                p_ = p[ i + skip : i + win + skip, e]
                C = np.polyfit(time_, p_, po)      
                C1 = np.dot(Deriv_mat, C)
                C2 = np.dot(Deriv_mat, C1)
    
                new_pos[i,e] = np.dot(ev_point, C)
                new_vel[i,e] = np.dot(ev_point, C1)*FPS
                new_acc[i,e] = np.dot(ev_point, C2)*FPS**2
                
                
    # smoothing the entire trajectory, not cutting the edges 
    elif cut_edges==False:
        for e,win,po in sequence:
            time_ = np.arange(win)
            ev_point = float(win/2)**np.arange(po + 1)[::-1]
            
            Deriv_mat = []
            for i in range(po+1):
                a = [0.0 for j in range(po+1)]
                if i!=0:
                    a[i-1] = po - (i-1)
                Deriv_mat.append(a)
            Deriv_mat = np.array(Deriv_mat)
            
            for i in range( new_N ):
                if i < win/2:          # get the portion of size widon for
                    p_ = p[:win, e]    # fitting the polynomial
                elif N - (1+i) < win/2:
                    p_ = p[-1*win:, e]
                else:
                    p_ = p[i-win/2 : i+win/2+1, e]
                
                C = np.polyfit(time_, p_, po)      
                C1 = np.dot(Deriv_mat, C)
                C2 = np.dot(Deriv_mat, C1)
                
                if i < win/2: 
                    ev_point = float(i%(win/2))**np.arange(po + 1)[::-1]
                elif N - (1+i) < win/2:
                    ev_point = float(win + i - N)**np.arange(po + 1)[::-1]
                else:
                    ev_point = float(win/2)**np.arange(po + 1)[::-1]
                
                new_pos[i,e] = np.dot(ev_point, C)
                new_vel[i,e] = np.dot(ev_point, C1)*FPS
                new_acc[i,e] = np.dot(ev_point, C2)*FPS**2
                
                
    
    if cut_edges:
        residuals = new_pos - p[W/2 : N-W/2,:]
        new_tr = Trajectory(new_pos, new_vel, traj.time()[W/2 : N-W/2],
        traj.trajid(), accel = new_acc, residuals = residuals)
    
    elif cut_edges==False : 
        residuals = new_pos - p
        new_tr = Trajectory(new_pos, new_vel, traj.time(),
        traj.trajid(), accel = new_acc, residuals = residuals)
    
    return new_tr









def smooth_traj(traj, FPS, window = 11, polyorder = 2, n = 2, cut_edges=True):
    '''
    this function's input is a flowtracks trajectory object. the output
    is a smoothed trajectory. for the smoothing a savizky golay filter
    is applied n times the position of the trajectory. the velocity
    and acceleration are calculated by a simple derivative. 
    
    if cut_edges==Ture - the returned trajectory will be shorter than the original 
    by 2*window since the begining and the end of the trajectory
    are trimed each by window.

    returns-
    new_traj - a smoothed trajectory object
    delta - the mean squared translation done by the smoothing
    '''
    if cut_edges:
    	if len(traj) < window*2 + 1:
        	raise ValueError('trajectory too short for window size %d with cutting'%window)
    else:
        if len(traj) < window + 1:
        	raise ValueError('trajectory too short for window size %d'%window)
        
    p = traj.pos()
    p_ = p 
    for i in range(n):
        p_ = sgf(p_, window, polyorder, axis = 0)
        v_ = np.gradient(p_)[0]*FPS
        a_ = np.gradient(p_)[0]*FPS
    
    delta = np.mean((p - p_)**2, axis = 0)

    if cut_edges:
        p_ = p_[window:-window, :]
        v_ = v_[window:-window, :]
        a_ = a_[window:-window, :]
        t_ = traj.time()[window:-window]
    else: 
        t_ = traj.time()
        new_traj = Trajectory(p_, v_, t_, traj.trajid(), accel = a_)
    return new_traj, delta








