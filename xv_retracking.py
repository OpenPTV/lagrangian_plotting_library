#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 15:12:36 2018

@author: ron

An implementation for a trajectory segments connection algorithm.
The algorithm is defined by Xu (2008): Haitao Xu, Tracking Lagrangian 
trajectories in position–velocity space, Meas. Sci. Technol. 19 (2008)
075105 (10pp) 
"""

import numpy as np
from smooth_trajectories import smooth_traj
from flowtracks.io import Trajectory
import matplotlib.pyplot as plt


class Traj_Conector(object):    
    '''
    This object connects pairs of broken trajectory segments 
    based on the algorithm of Xu(2008). 
    
    to use, after initiation, run the function -  
    Traj_Conector.make_new_traj_list()
    
    and then retrive the list of connected trajectories with -
    Traj_Conector.new_t 
    
    you can plot the new trajectories with -
    Traj_Conector.plot_connected_trajectories()
 
    '''
    
    def __init__(self, traj_list, Ts, dm, wa = 0, FPS = 1):
        '''
        traj_list - list of trajectory instances
        Ts - float maximum time to search (in seconds)
        dm - maximum distance for connection
        wa - scaling function for the acceleration
        FPS - frame rate (if trajectory.time() is in physical times
              use FPS = 1)
        '''
        self.t = traj_list
        self.Ts = Ts
        self.dm = dm
        self.wa = wa
        self.FPS = FPS
        self.dij = np.zeros( (len(self.t),len(self.t)) )
        
        
    def smooth_the_traj_list(self):
        '''
        the algorithm requires smoothing of the trajectories first
        in order to work properly. eventually, the connected trajectories
        are UNSMOOTHED. the smoothing is only used for the algorithm itself.
        
        smoothing is preformed with the smooth_traj function at the 
        load_traj_h5 module
        '''
        if self.FPS == 1: FPS = self.t[0].time()[1] - self.t[0].time()[0]
        else: FPS = self.FPS
        
        self.smoothed = []
        for tr in self.t:
            if len(tr) > 5:
                sm, d = smooth_traj(tr, FPS, 5, 2, n=1,  cut_edges=False)
            elif len(tr) > 3:
                sm, d = smooth_traj(tr, FPS, 3, 1, n=1,  cut_edges=False)
            else: 
                sm = tr
                
            self.smoothed.append(sm)


    def calc_d(self, i, j):
        '''
        for 2 trajectories i and j calculates the value dij. note 
        that trajectory i must come at later time than trajectory j. 
        
        i and j are indexes (int) for the trajectories i and j in the 
        full trajectory list (traj_list).
        '''
        dt = (self.smoothed[j].time()[0] - self.smoothed[i].time()[-1])/self.FPS
        if dt > self.Ts or dt<0:
            return 0
    
        xpi = self.smoothed[i].pos()[-1,:] + self.smoothed[i].velocity()[-1,:]*dt
        vpi = self.smoothed[i].velocity()[-1,:] + self.wa*self.smoothed[i].accel()[-1,:]*dt
        
        dx = np.linalg.norm(self.smoothed[j].pos()[0,:] - xpi)
        dv = np.linalg.norm(self.smoothed[j].velocity()[0,:] - vpi)
        
        d_ij = np.sqrt(dx**2 + (dv*dt)**2) 
        return d_ij * (d_ij <= self.dm)
    
    
    def calc_dij(self):
        '''
        fills in the values of d in the matrix d_ij
        '''
        N = len(self.t)
        for i in range(N):
            for j in range(i+1,N):
                self.dij[i,j] = self.calc_d(i, j)
    
    
    def list_traj(self):
        '''
        generate a list of trajectory indexes
        to connect to each other
        '''
        self.connect_list = []
        
        for i in range(len(self.t)):
            m = (-1,-1,self.dm)
            
            for j in range(len(self.dij[i,:])):
                if self.dij[i,j] < m[-1] and self.dij[i,j] != 0:
                    m = (i,j,self.dij[i,j])
            
            if m[-1] < self.dm:
                self.connect_list.append(m)

        
        self.connect_list = sorted(self.connect_list, key=lambda x:x[1])
        i = 0
        while i < len(self.connect_list)-1:
            check = 1
            if self.connect_list[i][1] == self.connect_list[i+1][1]:
                m=i
                if self.connect_list[i][2] < self.connect_list[i+1][2]:
                    m = i+1
                self.connect_list.remove(self.connect_list[m])
                check = 0
            if check:
                i += 1 
    
    def get_pairs(self):
        '''
        goes through all the steps 
        to construct the list of pairs to connect
        '''
        self.smooth_the_traj_list()
        self.calc_dij()
        self.list_traj()

    def connect_trajs(self, i, j):
        '''
        for 2 trajectory indexes will return a new trajectory instance
        that is stitched from the formers, plus an interpolated section 
        in the middle of them.
        '''
        return self.stitch(self.t[i], self.t[j])
        
        
    def stitch(self,tr1,tr2):
        t0 = tr1.time()[0]
        dt =  tr2.time()[1] - tr2.time()[0] 
        N_pnts = int((tr2.time()[0] - tr1.time()[-1])/dt) - 1
        new_time = [tm for tm in tr1.time()]
        added_time = []
        for i_ in range(N_pnts):
            added_time.append(new_time[-1] + dt)
            new_time.append(new_time[-1] + dt)
        new_time = np.append(new_time ,  tr2.time())
        
        
        if N_pnts == -1:
            new_time = np.append(tr1.time(), tr2.time()[1:])
            interped_pos = 0.5*(tr1.pos()[-1:,:] +tr2.pos()[0:1,:])
            new_pos = np.append(tr1.pos()[:-1,:] , interped_pos, axis=0)
            new_pos = np.append(new_pos , tr2.pos()[1:,:], axis=0)
        
        else:
            poly_2 = lambda C,x: np.dot(x**np.arange(2,-1,-1), C)
            tm_fit = np.append(tr1.time()[-3:] , tr2.time()[:3]) - t0
            
            try:
                interped_pos = np.zeros( (N_pnts,3) )
            except:
                print (N_pnts,3)
                print ('%d'%tr1.time()[-1])
                print ('%d'%tr2.time()[0])
                
            for i_ in range(3):
                x_fit = np.append(tr1.pos()[-3:, i_] , tr2.pos()[:3, i_])
                C = np.polyfit(tm_fit, x_fit, 2)
                for e,tm in enumerate(added_time):
                    interped_pos[e,i_] =  poly_2(C,tm - t0) 
            
            new_pos = np.append(tr1.pos() , interped_pos, axis=0)
            new_pos = np.append(new_pos , tr2.pos(), axis=0)
        
        if self.FPS == 1:
            new_vel = np.gradient(new_pos)[0] / dt
            new_accel = np.gradient(new_vel)[0] / dt
        else:
            new_vel = np.gradient(new_pos)[0] * self.FPS
            new_accel = np.gradient(new_vel)[0] * self.FPS
            
        '''
        new_pos = np.append(self.t[i].pos(), self.t[j].pos(), axis=0)
        new_vel = np.append(self.t[i].velocity(), self.t[j].velocity(), axis=0)
        new_accel = np.append(self.t[i].accel(), self.t[j].accel(), axis=0)
        new_time = np.append(self.t[i].time(), self.t[j].time())
        '''
        new_trid = tr1.trajid()
        new_traj = Trajectory(new_pos, new_vel, new_time,
                              new_trid, accel = new_accel)
        return new_traj
        
    
    def make_new_traj_list(self):
        '''
        will make a list of connected trajectories as well as trajectories
        that were not found a continuation.
        '''
        self.get_pairs()
        
        pairs = []
        for i in self.connect_list:
            pairs.append(i[0])
            pairs.append(i[1])
        
        self.new_t = []
        for i in range(len(self.t)):
            if i not in pairs:
                self.new_t.append(self.t[i])
        
        
        self.new_trajectories = []
        
        self.connect_list = sorted(self.connect_list,key = lambda a:a[0])
        to_connect = []
        for i in self.connect_list:
            chk = 0
            for lst in to_connect:
                if i[0] == lst[-1][1]:
                    lst.append(i)
                    chk = 1
            if chk == 0:
                to_connect.append([i])
            
        for lst in to_connect:
            for i in range(len(lst)):
                if i == 0:
                    new_traj = self.connect_trajs(lst[i][0],lst[i][1])
                else:
                    new_traj = self.stitch(new_traj, self.t[lst[i][1]])
            self.new_t.append(new_traj)
            self.new_trajectories.append(new_traj)
        
        # bug:
        #for i in self.connect_list:
        #    new_traj = self.connect_trajs(i[0],i[1])
        #    self.new_t.append(new_traj)
        #    self.new_trajectories.append(new_traj)
            
            
    def plot_connected_trajectories(self):
        fig, ax = plt.subplots(2,1, sharex=True)
        
        for tr in self.new_trajectories:
            ax[0].plot(tr.pos()[:,0], tr.pos()[:,1],'-x',color='r',ms=3,mew=1.5)
            ax[1].plot(tr.pos()[:,0], tr.pos()[:,2],'-x',color='r',ms=3,mew=1.5)
            
        for i in self.connect_list:
            tr = self.t[i[0]]
            ax[0].plot(tr.pos()[:,0], tr.pos()[:,1],'-o',color='b',ms=2)
            ax[1].plot(tr.pos()[:,0], tr.pos()[:,2],'-o',color='b',ms=3)
            tr = self.t[i[1]]
            ax[0].plot(tr.pos()[:,0], tr.pos()[:,1],'-o',color='b',ms=2)
            ax[1].plot(tr.pos()[:,0], tr.pos()[:,2],'-o',color='b',ms=2)
        
        ax[1].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[1].set_ylabel('z')
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        return fig, ax
        


def get_mean_traj_len(traj_list):
    '''
    returns the average lengths of trajectories in a list
    '''
    l = [tr.time()[-1]-tr.time()[0] for tr in traj_list]
    return np.mean(l)




def iterate_traj_connector(traj_list, N_max, Ts, dm, wa = 0, FPS = 1):
    '''
    running the traj connector over a very long list of trajectories
    is extremely inefficient. This function will iteratively subsample
    "N_max" trajectories and run the traj connector over the sub samples.
    
    returns - connected trajectory list
    '''
    t_ = sorted(traj_list, key = lambda tr:tr.time()[0])
    new_t = []
    for i in range(len(traj_list) / N_max):
        tc = Traj_Conector(t_[i*N_max:(i+1)*N_max],Ts, dm,wa=wa, FPS=FPS)
        tc.make_new_traj_list()
        for tr in tc.new_t:
            new_t.append(tr)
    
    return new_t
