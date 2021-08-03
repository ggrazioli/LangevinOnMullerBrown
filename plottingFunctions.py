'''
These functions can be used to generate various data visualizations of 
Langevin dynamics trajectoreis in two dimensions for a system
governed by a Muller-Brown potential. 

This code was written by: Gianmarc Grazioli on May 28, 2020

'''
# !/usr/bin/python
# -*- coding: utf8 -*-
# -------------------------------------------------------------------------
#
#   A simple molecular dynamics solver that simulates the motion
#   of 2D non-interacting particles in the canonical ensemble subjected
#   to a Muller-Brown potential using a Langevin thermostat.
#
# --------------------------------------------------------------------------
 
import numpy as np
import matplotlib.pylab as plt
import writeXYZ as xyz
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ArtistAnimation
from langevin_2D import getPotEnergy
from langevin_2D import getZvalues
    
def plotTemperatures(fname, tstep):
    tempOut = np.loadtxt(fname)
    plt.plot(tempOut[:,0] * tstep, tempOut[:,1], marker = 'o', markersize=1)
    plt.xlabel('Time (ps)')
    plt.ylabel('Temperature (K)')
    plt.show()

def getTrajMovie(traj, box):
    #Get mesh of points for the countour plot
    x = np.linspace(box[0][0], box[0][1], 33)
    y = np.linspace(box[1][0], box[1][1], 33)
    X, Y = np.meshgrid(x, y)
    Z = getPotEnergy(X, Y)
    #Set ticks for depth range of contour plot
    cbarticks = np.linspace(min(np.ravel(Z)), 1., 20)
    #Initialize figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(xlim=(box[0]), ylim=(box[1]))
    points, = ax.plot([], [], 'o', color=(.75,1,1))
    #Create contour plot
    image = ax.contourf(X, Y, Z, cbarticks, vmax=1.)
    plt.colorbar(image, ax=ax,ticks=cbarticks)
    
    def init():
        points.set_data([], [])
        return points,
    def animate(i):
        x = traj[i][:,0]
        y = traj[i][:,1]
        points.set_data(x, y)
        return points,
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=len(traj), interval=75, blit=True)
    anim.save('traj.gif', writer='pillow')
    return anim

def getMovieWithTemp(traj, box, temperatureFile):
    temps = np.loadtxt(temperatureFile)
    tempStep = round(len(temps)/len(traj))
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Temperature (K)')
    
    #Get mesh of points for the countour plot
    x = np.linspace(box[0][0], box[0][1], 33)
    y = np.linspace(box[1][0], box[1][1], 33)
    X, Y = np.meshgrid(x, y)
    Z = getPotEnergy(X, Y)
    #Set ticks for depth range of contour plot
    cbarticks = np.linspace(min(np.ravel(Z)), 1., 20)
    ax2.contourf(X, Y, Z, cbarticks, vmax=1.)
    
    lines = []
    for i in range(1, len(traj)):
        line1,  = ax1.plot(temps[0:(i*tempStep),0], temps[0:(i*tempStep),1], color='blue')
        xTraj = traj[i][:,0]
        yTraj = traj[i][:,1]
        line2, = ax2.plot(xTraj, yTraj, 'o', color=(.75,1,1))
        lines.append([line1, line2])
    anim = ArtistAnimation(fig,lines,interval=75,blit=True)
    return anim

def getSurfaceForPymol(box, stepsInX, stepsInY, fname, scaling=.006, zCutoff=.1):
    # Make new file then close to prevent appending to previous session data
    with open(fname, 'w') as fp:
        fp.close()
    x = np.linspace(box[0][0], box[0][1], stepsInX)
    y = np.linspace(box[1][0], box[1][1], stepsInY)
    xx, yy = np.meshgrid(x, y, sparse = False)
    z = getPotEnergy(xx, yy)*scaling
    pointList = np.vstack([np.ravel(xx), np.ravel(yy), np.ravel(z)]).T
    pointList = np.array([i for i in pointList if i[2]<zCutoff]) 
    xyz.writeXYZ(fname, len(pointList), 0, atomType=np.array(["H"]*len(pointList)),  pos=pointList)

def writeTrajXYZ(traj, ofname):
    # Make new file then close to prevent appending to previous session data
    with open(ofname, 'w') as fp:
        fp.close()
    scaleForPymol = .006 #Scaling for Pymol plotting
    natoms = traj.shape[1]
    atomType = np.array(["H"]*traj.shape[1])
    for i in range(0, len(traj)):
        pos3D = getZvalues(traj[i], scaleForPymol)
        xyz.writeXYZ(ofname, natoms, i, atomType=atomType, pos=pos3D)
