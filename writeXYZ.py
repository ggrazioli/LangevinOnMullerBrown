'''
Created on May 29, 2020 by Gianmarc Grazioli

This code is used to produce (x,y,z) files from Muller-Brown simulations. 
It can be used to output a mesh of "hydrogen atoms," which can be used 
to create 3D renderings potential energy surfaces in molecular dynamics 
software visualization tools like VMD and Pymol. It can also be used to 
output the positions of all particles moving on a Muller-Brown potential. 
Here is an example of such a visualization in action: 
https://youtu.be/BsEVmInG3Oo   

This function is based on one created on September 22, 2018
by Andrew Abi-Mansour
'''

# !/usr/bin/python
# -*- coding: utf8 -*-


import numpy as np

def writeXYZ(filename, natoms, timestep, **data):
    """ Writes the output (in .xyz format) """
    
    axis = ('x', 'y', 'z')
    
    with open(filename, 'a') as fp:
        
        fp.write('{}\n'.format(natoms))
            
        keys = list(data.keys())
        
        for key in keys:
            isMatrix = len(data[key].shape) > 1
            
            if isMatrix:
                _, nCols = data[key].shape
                
                for i in range(nCols):
                    if key == 'pos':
                        data['{}'.format(axis[i])] = data[key][:,i]
                    else:
                        data['{}_{}'.format(key,axis[i])] = data[key][:,i]
                        
                del data[key]
                
        keys = data.keys()
        
        output = []
        for key in keys:
            output = np.hstack((output, data[key]))
            
        if len(output):
            np.savetxt(fp, output.reshape((natoms, len(data)), order='F'), fmt="%s", delimiter="   ")# was fmt="%s"
