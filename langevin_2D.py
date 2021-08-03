'''
This code can be used to run Langevin dynamics in two dimensions for a system
governed by a Muller-Brown potential. This code was written
by: Gianmarc Grazioli on May 28, 2020

The structure of this code is based on a 3D Langevin dynamics code for 
free particles, which was created on September 22, 2018 by Andrew Abi-Mansour

The 3D code by Andrew Abi-Mansour can be found here: 
https://github.com/Comp-science-engineering/Tutorials/tree/master/MolecularDynamics
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

# Define global physical constants
Avogadro = 6.02214086e23
Boltzmann = 1.38064852e-23
    
# This function takes a position of a 2D particle as input and 
# returns the force on the particle due to a Mueller-Brown potential 
def potForce(pos):
    """
    Parameters
    ----------
    pos : np.array() or list
        position of particles in 2D space.

    Returns
    -------
    np.array()
        force vector on particle due to Muller-Brown potential.
    """
    def getOnePointForce(p):
        a = np.empty(2) #acceleration array initialized
        a[0] = (-400*np.exp(-(-1 + p[0])**2 - 10*p[1]**2)*(-1 + p[0]) - 
                200*np.exp(-p[0]**2 - 10*(-0.5 + p[1])**2)*p[0] + 
                170*np.exp(-6.5*(0.5 + p[0])**2 + 11*(0.5 + p[0])*(-1.5 + p[1]) - 
                6.5*(-1.5 + p[1])**2)*(-13.*(0.5 + p[0]) + 11*(-1.5 + p[1])) - 
                15*np.exp(0.7*(1 + p[0])**2 + 0.6*(1 + p[0])*(-1 + p[1]) + 0.7*(-1 + 
                p[1])**2)*(1.4*(1 + p[0]) + 0.6*(-1 + p[1])))
              
        a[1] = (170*np.exp(-6.5*(0.5 + p[0])**2 + 11*(0.5 + p[0])*(-1.5 + p[1]) - 
                6.5*(-1.5 + p[1])**2)*(11*(0.5 + p[0]) - 13.*(-1.5 + p[1])) - 
                15*np.exp(0.7*(1 + p[0])**2 + 0.6*(1 + p[0])*(-1 + p[1]) + 0.7*(-1 + 
                p[1])**2)*(0.6*(1 + p[0]) + 1.4*(-1 + p[1])) - 2000*np.exp(-p[0]**2 - 
                10*(-0.5 + p[1])**2)*(-0.5 + p[1]) - 4000*np.exp(-(-1 + p[0])**2 - 
                10*p[1]**2)*p[1])
        return a
    return np.apply_along_axis(getOnePointForce, 1, pos)

def getPotEnergy(x, y):
    """
    Calculates potential energy as a function of position for Muller-Brown

    Parameters
    ----------
    p : a point in 2D space (np.array() or list)

    Returns
    -------
        float
        Potential energy at point p due to Muller-Brown potential.
    """
    z = (-170*np.exp(-6.5*(0.5 + x)**2 + 11*(0.5 + x)*(-1.5 + y) - 
            6.5*(-1.5 + y)**2) + 15*np.exp(0.7*(1 + x)**2 + 0.6*(1 + 
            x)*(-1 + y) + 0.7*(-1 + y)**2) - 100*np.exp(-x**2 - 
            10*(-0.5 + y)**2) - 200*np.exp(-(-1 + x)**2 - 10*y**2))
    return z

def getZvalues(pos, scaling):
    """
    Function to get scaled potential energy as the z-axis for all points in pos

    Parameters
    ----------
    pos : list
        list of positions.
    scaling : float
        Constant scaling of potential energy for plotting purposes.

    Returns
    -------
    list
        positions and scaled energy values.

    """
    zList = getPotEnergy(pos[:,0], pos[:,1])
    return np.column_stack([pos, scaling*zList])


def computeForce(mass, vels, temp, relax, dt, pos):
    """ Computes the Langevin force for all particles

    @mass: particle mass (ndarray)
    @vels: particle velocities (ndarray)
    @temp: temperature (float)
    @relax: thermostat constant (float)
    @dt: simulation timestep (float)
    @pos: position in space

    returns forces (ndarray)
    """

    natoms, ndims = vels.shape
    sigma = np.sqrt(2.0 * mass * temp * Boltzmann / (relax * dt))
    noise = np.random.randn(natoms, ndims) * sigma[np.newaxis].T
    #force = dissipative force + random kicks + negative gradient of the potential
    force = - (vels * mass[np.newaxis].T) / relax + noise + potForce(pos)
    return force

def integrate(pos, vels, forces, mass,  dt):
    """ Evolve the system in time by one step using an Euler integrator 

    pos: atomic positions (ndarray, updated)
    vels: atomic velocity (ndarray, updated)
    """
    
    pos += vels * dt
    vels += forces * dt / mass[np.newaxis].T

def run(**args):
    """ This is the main function that solves Langevin's equations for
    a system of natoms using a forward Euler scheme, and returns a trajectory
    and plots temperature as a function of time.
    
    @natoms (int): number of particles
    @temp (float): temperature (in Kelvin)
    @mass (float): particle mass (in Kg)
    @relax (float): relaxation constant (in seconds)
    @dt (float): simulation timestep (s)
    @nsteps (int): total number of steps the solver performs
    @box (tuple): simulation box size (in meters) of size dimensions x 2
    e.g. box = ((-1e-9, 1e-9), (-1e-9, 1e-9)) defines a 2D square
    @ofname (string): filename to write output to
    @freq (int): write output every 'freq' steps
    
    Returns the trajectory as an np.ndarray of shape [timeSteps, natoms, dimensions].
    
    """

    natoms, box, dt, temp = args['natoms'], args['box'], args['dt'], args['temp']
    mass, relax, nsteps   = args['mass'], args['relax'], args['steps']
    initPos, freq = args['initPos'], args['freq']
    
    dim = len(box)
    pos = np.vstack((np.random.normal(initPos[0], .1, natoms), np.random.normal(initPos[1], .1, natoms))).T
    vels = np.random.rand(natoms,dim)
    mass = np.ones(natoms) * mass / Avogadro
    step = 0
    tempOut = []
    traj = [pos]
    
    while step <= nsteps:

        step += 1

        # Compute all forces
        forces = computeForce(mass, vels, temp, relax, dt, pos)

        # Move the system in time
        integrate(pos, vels, forces, mass, dt)

        # Compute output (temperature)
        ins_temp = np.sum(np.dot(mass, (vels - vels.mean(axis=0))**2)) / (Boltzmann * dim * natoms)
        tempOut.append([step * dt, ins_temp])
        
        if not step%freq:
            traj = np.vstack((traj, [pos]))
        
        if step%1000 == 0:
            print("Running step",step,"of",nsteps,"steps")
    print("Dynamics complete!")        
    tempOut = np.array(tempOut)
    np.savetxt("temperatures.txt", tempOut)
    return traj

if __name__ == '__main__':

    params = {
        'natoms': 50,
        'temp': 400, #was 300
        'mass': 0.001,
        'relax': 1e-13,
        'dt': 1e-17, #was 1e-15
        'steps': 2000,
        'freq': 10,
        'box': ((-1.7, 1.2), (-.7, 2.2)), 
        'initPos': (-.75, .6),
        }
    traj = run(**params)
    