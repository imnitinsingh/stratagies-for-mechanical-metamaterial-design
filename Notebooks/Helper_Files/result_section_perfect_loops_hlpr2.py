# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:47:24 2015

@author: singh
"""

# ------------------------------------------------IMPORT ALL THE PACKAGES----------------------------------------------------------------------------
import numpy as np ;    import matplotlib ;    from matplotlib import pyplot as plt ;   from matplotlib.widgets import Slider,  Button,  RadioButtons 
import pylab ;     import matplotlib.patches as patches;     import random; import scipy ;   import math ;   import sympy as sp ; from sympy import * 
from scipy.spatial import distance ; import time; start_time = time.time()
# ---------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------THESE NOTATIONS MAKE THE WRITIGN EASIER---------------------------------------------------------------------
pi = np.pi ;dot = np.dot ; sin = np.sin ; cos = np.cos ; ar = np.array; sqrt = np.sqrt; rand = scipy.rand ; arange = scipy.arange ; show = pylab.show
plot = pylab.plot ;  axis = pylab.axis ; grid = pylab.grid ;  title  = pylab.title ; atan = np.arctan; transpose = np.transpose ; dotProduct = np.dot
# ---------------------------------------------------------------------------------------------------------------------------------------------------

'''
#--------------------------------------FUNCTION CALLED FROM visualize FUNCTION TO RETURN COLORED ANGLE PATCHES---------------------------------------
def get_angle_plot(l1xy, l2xy, i, lw, offset, origin = [],  len_x_axis = 1, len_y_axis = 1):

    angle1 = np.degrees(np.arctan2(float(l1xy[1][1] - l1xy[0][1]) , float(l1xy[1][0] - l1xy[0][0]))) 

    angle2 = np.degrees(np.arctan2((l2xy[1][1] - l2xy[0][1]) , float(l2xy[1][0] - l2xy[0][0])))

    color = ['red', 'green', 'blue', 'yellow', 'black']
    return patches.Arc(origin, len_x_axis*offset, len_y_axis*offset, 0, angle2, angle1, lw = lw, color=color[i], label = str(angle1-angle2)+u"\u00b0")
#----------------------------------------------------------------------------------------------------------------------------------------------------    
'''

#---------------------------------------------MAIN VISUALIZATION FUNCTION----------------------------------------------------------------------------  

def visualize(position_vector, n, shape_info_bodies, clr_, alpha_ = 1):
    corner_body_array = np.zeros((8,4,2))
    for i in range(0, n**2-1):
               
        corner_body = np.array([[l*np.cos(theta+position_vector[2+3*i,0])+position_vector[0+3*i,0], l*np.sin(theta+position_vector[2+3*i,0])+\
                                                                    position_vector[1+3*i,0]] for l,theta in zip(shape_info_bodies[i, :, 0],shape_info_bodies[i, :, 1])]) 
          
        corner_body_array[i,:,:] = corner_body
    return(corner_body_array)        
