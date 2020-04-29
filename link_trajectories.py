# -*- coding: utf-8 -*-
"""
Problem: the candidate may already be linked. Need to register the candidate 
and update both if the candidate has a better link.

Created on Wed Feb 22 12:08:02 2017

@author: yosef, based on attempt by lillyverso
"""

import itertools as it, numpy as np
import matplotlib.pyplot as plt

from flowtracks.scene import Scene
from flowtracks.trajectory import Trajectory

# Unpack the param
inName = 'data/ptv_scene16_testYosef/lillys_particles.h5'
frate = 100
scn = Scene(inName)
# Can limit the frame range but not a must:
# scn.set_frame_range((param.first, param.last))

## Hitao linking criteria -----------------------------------------------------
#distThresh=0.005;   
#maxDt=1;
#minLength=3;  

distThresh = 0.003;   
maxDt = 0.025;
minLength = 3;       # length of elemnts in the trajs

# Though this won't be necessary if you don't save the short ones at all in the
# HDF conversion.
long_trajects = it.ifilter(
    lambda trj: len(trj) > minLength, scn.iter_trajectories())

# Keyed by trajid, value is a tuple (id, dist) where id is the best candidate 
# trajectory for linking, and dist is the average-distance measure for this
# pair (the measure to beat)
links = {}
back_links = {}

for trj1, trj2 in it.combinations(long_trajects, 2):
    dt = (trj2.time(0) - trj1.time(-1))/frate
    if not (0 < dt <= maxDt):
        continue
    
    master_id = trj1.trajid()
    slave_id = trj2.trajid()
    links.setdefault(master_id, (None, distThresh))
    back_links.setdefault(slave_id, (None, distThresh))
    min_dist = min(links[master_id][1], back_links[slave_id][1])
    
    # Continue trj1 forward one time interval, and trj2 backward one interval.
    # If the evarage distance between each predicted point and the other traj's
    # endpoint meets the criteria - connect.
    predicted_forward = trj1.pos(-1) + dt*trj1.velocity(-1)
    predicted_backward = trj2.pos(0) - dt*trj2.velocity(0)
    dist_forward = np.linalg.norm(predicted_forward - trj2.pos(0))
    dist_backward = np.linalg.norm(predicted_backward - trj1.pos(-1))
    
    # Possible register candidate:
    avg_dist = (dist_forward + dist_backward)/2.
    if avg_dist < min_dist:
        old_link = back_links[slave_id][0]
        if old_link is not None:
            links[old_link] = (None, distThresh)
        links[master_id] = (slave_id, avg_dist)
        back_links[slave_id] = (master_id, avg_dist)
 
print(("candidates:", links))

# Weld the final best candidates.
out_trajects = []
used_trids = set() # don't repeat taken candidates as masters.
for trid, cand in list(links.items()):
    if trid in used_trids:
        continue
    
    trj_weld = scn.trajectory_by_id(trid)
    while cand[0] is not None:
        used_trids.add(cand[0])
        trj1 = trj_weld
        trj2 = scn.trajectory_by_id(cand[0])
        trj_weld = Trajectory(
            np.vstack((trj1.pos(), trj2.pos())),
            np.vstack((trj1.velocity(), trj2.velocity())),
            trajid=trj1.trajid(),
            time=np.hstack((trj1.time(), trj2.time())),
            accel=np.vstack((trj1.accel(), trj2.accel())),
        )
        
        if cand[0] not in links:
            break
        cand = links[cand[0]]
        
    out_trajects.append(trj_weld)

# Check wheter we link correctly the trajs
# plot the trajs
fig = plt.figure(figsize=(7,7))
for trj in out_trajects:
    pos = trj.pos()
    plt.plot(pos[:,0], pos[:,1])

plt.show()