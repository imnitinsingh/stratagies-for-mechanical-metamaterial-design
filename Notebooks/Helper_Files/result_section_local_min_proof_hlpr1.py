# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:38:37 2016

@author: singh
"""



# ----------------------------------------------------USED PACKAGES-----------------------------------------------------
import numpy as np;import matplotlib;from matplotlib import pyplot as plt;import pylab;from scipy.spatial import distance       
import matplotlib.patches as patches;import random; import scipy; import math; import sympy as sp#; from sympy import  * 
from scipy.spatial  import distance ; import time; start_time = time.time(); import os; import profile
from math import atan2
# ----------------------------------------------------------------------------------------------------------------------


#----------------------------------------------THESE NOTATIONS MAKE THE WRITIGN EASIER----------------------------------
pi = np.pi ;   dot = np.dot ;    sin = np.sin ;    cos = np.cos ;    ar = np.array ;  sqrt = np.sqrt; rand = scipy.rand  
arange = scipy.arange; show = pylab.show; plot = pylab.plot; axis = pylab.axis; grid = pylab.grid; title  = pylab.title 
atan = np.arctan;transpose = np.transpose;dotProduct = np.dot;conc = np.concatenate;lnsp = np.linspace; log10 = np.log10
dist = np.linalg.norm
#-----------------------------------------------------------------------------------------------------------------------

#plt.close('all')

#--------------------------------------------------------------OOP PART-------------------------------------------------
class UpdateNetwork(object):
    
    
    def __init__(self, X1, Y1, Theta1, LenVec1, Phi1, X2, Y2, Theta2, LenVec2, Phi2, BCPM2D, BCPM3D): 
        """Initialize the network by defining the current position vector, the connectivity among the members, the shape
        information of the members. Preallocate the First Derivative Vector(FDV) and the  Second  Derivative Matrix(SDM) 
        for the network."""
        
        self.X1 = X1; self.Y1 = Y1; self.Theta1 = Theta1; self.LenVec1 = LenVec1; self.Phi1 = Phi1
        self.X2 = X2; self.Y2 = Y2; self.Theta2 = Theta2; self.LenVec2 = LenVec2; self.Phi2 = Phi2       
        self.BCPM2D = BCPM2D
        self.BCPM3D = BCPM3D        
        self.FDV  = np.zeros((10, 6))
        self.SDM  = np.zeros((10, 6, 6))      
        
    
    def UpdateFirstDerivativeVector(self):
        """Fill  the appropriate elements of FDV by considering every connection in the network  one by one. Derivatives 
        are calculated analytically."""
        
        X1 = self.X1; Y1 = self.Y1; Theta1 = self.Theta1; LenVec1 = self.LenVec1; Phi1 = self.Phi1  
        X2 = self.X2; Y2 = self.Y2; Theta2 = self.Theta2; LenVec2 = self.LenVec2; Phi2 = self.Phi2          
            
        self.FDV[:, 0]     =  2*X1 - 2*X2 + 2*LenVec1*cos(Theta1 + Phi1) - 2*LenVec2*cos(Theta2 + Phi2)
                                                                           
        self.FDV[:, 1]     =  2*Y1 - 2*Y2 + 2*LenVec1*sin(Theta1 + Phi1) - 2*LenVec2*sin(Theta2 + Phi2)
        
        self.FDV[:, 2]     =  2*LenVec1*(-X1 + X2 - LenVec1*cos(Theta1 + Phi1) + LenVec2*cos(Theta2 + Phi2))\
                                                *sin(Theta1 + Phi1) - 2*LenVec1*(-Y1 + Y2 - LenVec1*sin(Theta1 + Phi1) +\
                                                LenVec2*sin(Theta2 + Phi2))*cos(Theta1 + Phi1)
        
        self.FDV[:, 3]     =  -2*X1 + 2*X2 - 2*LenVec1*cos(Theta1 + Phi1) + 2*LenVec2*cos(Theta2 + Phi2)
        
        self.FDV[:, 4]     =  -2*Y1 + 2*Y2 - 2*LenVec1*sin(Theta1 + Phi1) + 2*LenVec2*sin(Theta2 + Phi2)
        
        self.FDV[:, 5]     =  -2*LenVec2*(-X1 + X2 - LenVec1*cos(Theta1 + Phi1) + LenVec2*cos(Theta2 + Phi2))\
                                                *sin(Theta2 + Phi2) + 2*LenVec2*(-Y1 + Y2 - LenVec1*sin(Theta1 + Phi1) +\
                                                LenVec2*sin(Theta2 + Phi2))*cos(Theta2 + Phi2) 
        
        self.FDV = self.FDV.flatten()[(np.cumsum(self.BCPM2D).reshape(self.BCPM2D.shape)-1)*self.BCPM2D] * self.BCPM2D
        self.FDV = np.sum(self.FDV, axis = 0)  
        self.FDV = self.FDV.reshape(24,1)
        
        self.FDV[5,0] = 0.0
        self.FDV[14,0] = 0.0
        self.FDV[12,0] = 0.0; self.FDV[13,0] = 0.0 
        return self.FDV 
     
     
    def UpdateSecondDerivativeMatrix(self):
        """Fill the appropriate elements of SDM by considering every connection in the network one by one. Derivatives 
        are calculated analytically."""
        X1 = self.X1; Y1 = self.Y1; Theta1 = self.Theta1; LenVec1 = self.LenVec1; Phi1 = self.Phi1  
        X2 = self.X2; Y2 = self.Y2; Theta2 = self.Theta2; LenVec2 = self.LenVec2; Phi2 = self.Phi2
        
        self.SDM[:,0,0]   +=  2.0 
        self.SDM[:,1,1]   +=  2.0
        self.SDM[:,3,3]   +=  2.0
        self.SDM[:,4,4]   +=  2.0
        
        self.SDM[:,0,3]   += -2.0               
        self.SDM[:,3,0]   += -2.0
        self.SDM[:,1,4]   += -2.0 
        self.SDM[:,4,1]   += -2.0
        
        self.SDM[:,0,1] = self.SDM[:,1,0] = self.SDM[:,0,4] = self.SDM[:,4,0] = self.SDM[:,1,3] = self.SDM[:,3,1] = \
        self.SDM[:,3,4] = self.SDM[:,4,3] = 0.0
                                    
        self.SDM[:,0,2]   += -2*LenVec1*sin(Theta1 + Phi1)
        self.SDM[:,2,0]   += -2*LenVec1*sin(Theta1 + Phi1) 
        self.SDM[:,0,5]   +=  2*LenVec2*sin(Theta2 + Phi2)
        self.SDM[:,5,0]   +=  2*LenVec2*sin(Theta2 + Phi2)
        self.SDM[:,1,2]   +=  2*LenVec1*cos(Theta1 + Phi1)
        self.SDM[:,2,1]   +=  2*LenVec1*cos(Theta1 + Phi1) 
        self.SDM[:,1,5]   += -2*LenVec2*cos(Theta2 + Phi2)
        self.SDM[:,5,1]   += -2*LenVec2*cos(Theta2 + Phi2) 
        
        self.SDM[:,2,2]    +=   2*LenVec1*(LenVec1*sin(Theta1 + Phi1)**2 + LenVec1*cos(Theta1 + Phi1)**2 - (X1 - X2 + \
                                LenVec1*cos(Theta1 + Phi1)- LenVec2*cos(Theta2 + Phi2))*cos(Theta1 + Phi1) - (Y1 - Y2 +\
                                LenVec1*sin(Theta1 + Phi1) - LenVec2*sin(Theta2 + Phi2))*sin(Theta1 + Phi1))
        
        self.SDM[:,2,3]    +=  2*LenVec1*sin(Theta1 + Phi1)
        self.SDM[:,3,2]    +=  2*LenVec1*sin(Theta1 + Phi1) 
        self.SDM[:,2,4]    += -2*LenVec1*cos(Theta1 + Phi1)
        self.SDM[:,4,2]    += -2*LenVec1*cos(Theta1 + Phi1)
        
        self.SDM[:,2,5]    += -2*LenVec1*LenVec2*(sin(Theta1+Phi1)*sin(Theta2+Phi2) + cos(Theta1+Phi1)*cos(Theta2+Phi2))
        self.SDM[:,5,2]    += -2*LenVec1*LenVec2*(sin(Theta1+Phi1)*sin(Theta2+Phi2) + cos(Theta1+Phi1)*cos(Theta2+Phi2))
        
        self.SDM[:,3,5]    += -2*LenVec2*sin(Theta2 + Phi2) 
        self.SDM[:,5,3]    += -2*LenVec2*sin(Theta2 + Phi2) 
        self.SDM[:,4,5]    +=  2*LenVec2*cos(Theta2 + Phi2)
        self.SDM[:,5,4]    +=  2*LenVec2*cos(Theta2 + Phi2)
        
        self.SDM[:,5,5]    +=  2*LenVec2*(LenVec2*sin(Theta2 + Phi2)**2 + LenVec2*cos(Theta2 + Phi2)**2 + (X1 - X2 +\
                                 LenVec1*cos(Theta1 + Phi1)- LenVec2*cos(Theta2 + Phi2))*cos(Theta2 + Phi2) + (Y1 - Y2\
                                 + LenVec1*sin(Theta1 + Phi1) -LenVec2*sin(Theta2 + Phi2))*sin(Theta2 + Phi2))
                                  
        self.SDM = self.SDM.flatten()[(np.cumsum(self.BCPM3D).reshape(self.BCPM3D.shape)-1)*self.BCPM3D] * self.BCPM3D  
        self.SDM = np.sum(self.SDM, axis = 0)
        
        self.SDM[5, :] = self.SDM[:, 5] = 0.0
        self.SDM[14, :] = self.SDM[:, 14] = 0.0
        self.SDM[12, :] = self.SDM[:, 12] = 0.0; self.SDM[13, :] = self.SDM[:, 13] = 0.0 
        return self.SDM


    def UpdateEnergyInSprings(self):
        X1 = self.X1; Y1 = self.Y1; Theta1 = self.Theta1; LenVec1 = self.LenVec1; Phi1 = self.Phi1  
        X2 = self.X2; Y2 = self.Y2; Theta2 = self.Theta2; LenVec2 = self.LenVec2; Phi2 = self.Phi2
        
        self.Energy = ((LenVec1*np.cos(Phi1+Theta1)+X1)-(LenVec2*np.cos(Phi2+Theta2)+X2))**2+((LenVec1*np.sin(Phi1+\
                               Theta1)+Y1)-(LenVec2*np.sin(Phi2+Theta2)+Y2))**2
        self.Energy = np.sum(self.Energy, axis = 0)                
        return self.Energy                        
#-----------------------------------------------------------------------------------------------------------------------
         
         


def hlpr_fun(pts):
    #valConstraint2 = 0; valConstraint3 = 0        
    n = 3 ; numberOfPolygons = n**2
    numberOfConnections =  int((2*4  +  3*4*(n-2)  +  4*(n**2 - 4*(n-2) -4))/2) 
    
    #-----------------------------------INITIALIZE THE POSITION VECTOR OF THE PARTICLES-------------------------------------
    '''SPV encodes the position and orientation of the polygons. Preallocate the position vector. Fill in 
    the appropriate places of the vector and randomize things a bit.'''                                
    SPV = np.zeros((3*(numberOfPolygons - 1), 1), dtype = float)   
                                             
    
    SPV[0],  SPV[1]  = (pts[0]+pts[1])/2;    SPV[2] = 0
    SPV[3],  SPV[4]  = (pts[1]+pts[2]+pts[9])/3;  SPV[5] = 0                            
    SPV[6],  SPV[7]  = (pts[2]+pts[3])/2;  SPV[8] = 0
    SPV[9],  SPV[10] = (pts[0]+pts[7]+pts[8])/3; SPV[11] = 0    
    SPV[12], SPV[13] = (pts[8]+pts[9]+pts[10]+pts[11])/4; SPV[14] = 0    
    SPV[15], SPV[16] = (pts[3]+pts[4]+pts[10])/3; SPV[17] = 0 
    SPV[18], SPV[19] = (pts[6]+pts[7])/2; SPV[20] = 0
    SPV[21], SPV[22] = (pts[5]+pts[6]+pts[11])/3;  SPV[23] = 0
    #-----------------------------------------------------------------------------------------------------------------------
        
    

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #------------------------------------ENCODE THE SHAPE INFO INTO ALL THE PARTICLES---------------------------------------
    """Fill the Shape Info in a np.array with shape (n**2, 4, 2) and randomize things a bit"""
        
    ShapeInfo = np.zeros((numberOfPolygons - 1, 4, 2), dtype = float) 
      
    #--------------------------------------TILE 0---------------------------------------------------------------------------  
    ShapeInfo[0, 0, 0]  = dist(pts[1] - pts[0])/2; ShapeInfo[0, 1, 0] = dist(pts[1] - pts[0])/2
    ShapeInfo[0, 2, 0]  = dist(pts[1] - pts[0])/2; ShapeInfo[0, 3, 0] = dist(pts[1] - pts[0])/2 
    ShapeInfo[0, 0, 1]  = atan2(pts[1,1] - SPV[1], pts[1,0] - SPV[0]) if atan2(pts[1,1] - SPV[1], pts[1,0] - SPV[0]) > 0 \
                          else 2*np.pi + atan2(pts[1,1] - SPV[1], pts[1,0] - SPV[0])  
    ShapeInfo[0, 1, 1] = atan2(pts[0,1] - SPV[1], pts[0,0] - SPV[0]) if atan2(pts[0,1] - SPV[1], pts[0,0] - SPV[0]) > 0 \
                          else 2*np.pi + atan2(pts[0,1] - SPV[1], pts[0,0] - SPV[0]) 
    ShapeInfo[0, 2, 1] = ShapeInfo[0, 1, 1]; ShapeInfo[0, 3, 1] =  ShapeInfo[0, 0, 1]                    
                    
    #-----------------------------------------------------------------------------------------------------------------------
                    
    #--------------------------------------TILE 1---------------------------------------------------------------------------                
    ShapeInfo[1, 0, 0] = dist([pts[1,0]-SPV[3], pts[1,1]-SPV[4]]) 
    ShapeInfo[1, 1, 0] = dist([pts[1,0]-SPV[3], pts[1,1]-SPV[4]])
    ShapeInfo[1, 2, 0] = dist([pts[9,0]-SPV[3], pts[9,1]-SPV[4]])
    ShapeInfo[1, 3, 0] = dist([pts[2,0]-SPV[3], pts[2,1]-SPV[4]])
    
    ShapeInfo[1, 0, 1] = atan2(pts[1,1] - SPV[4], pts[1,0] - SPV[3]) if atan2(pts[1,1] - SPV[4], pts[1,0] - SPV[3]) > 0 \
                          else 2*np.pi + atan2(pts[1,1] - SPV[4], pts[1,0] - SPV[3])
    ShapeInfo[1, 1, 1] = atan2(pts[1,1] - SPV[4], pts[1,0] - SPV[3]) if atan2(pts[1,1] - SPV[4], pts[1,0] - SPV[3]) > 0 \
                          else 2*np.pi + atan2(pts[1,1] - SPV[4], pts[1,0] - SPV[3])
    ShapeInfo[1, 2, 1] = atan2(pts[9,1] - SPV[4], pts[9,0] - SPV[3]) if atan2(pts[9,1] - SPV[4], pts[9,0] - SPV[3]) > 0 \
                          else 2*np.pi + atan2(pts[9,1] - SPV[4], pts[9,0] - SPV[3])
    ShapeInfo[1, 3, 1] = atan2(pts[2,1] - SPV[4], pts[2,0] - SPV[3]) if atan2(pts[2,1] - SPV[4], pts[2,0] - SPV[3]) > 0 \
                          else 2*np.pi + atan2(pts[2,1] - SPV[4], pts[2,0] - SPV[3])
    #-----------------------------------------------------------------------------------------------------------------------
    
    #--------------------------------------TILE 2---------------------------------------------------------------------------
    ShapeInfo[2, 0, 0] = dist([pts[2,0]-SPV[6], pts[2,1]-SPV[7]]) 
    ShapeInfo[2, 1, 0] = dist([pts[2,0]-SPV[6], pts[2,1]-SPV[7]])
    ShapeInfo[2, 2, 0] = dist([pts[3,0]-SPV[6], pts[3,1]-SPV[7]])
    ShapeInfo[2, 3, 0] = dist([pts[3,0]-SPV[6], pts[3,1]-SPV[7]])
    
    ShapeInfo[2, 0, 1] = atan2(pts[2,1] - SPV[7], pts[2,0] - SPV[6]) if atan2(pts[2,1] - SPV[7], pts[2,0] - SPV[6]) > 0 \
                          else 2*np.pi + atan2(pts[2,1] - SPV[7], pts[2,0] - SPV[6])
    ShapeInfo[2, 1, 1] = atan2(pts[2,1] - SPV[7], pts[2,0] - SPV[6]) if atan2(pts[2,1] - SPV[7], pts[2,0] - SPV[6]) > 0 \
                          else 2*np.pi + atan2(pts[2,1] - SPV[7], pts[2,0] - SPV[6])
    ShapeInfo[2, 2, 1] = atan2(pts[3,1] - SPV[7], pts[3,0] - SPV[6]) if atan2(pts[3,1] - SPV[7], pts[3,0] - SPV[6]) > 0 \
                          else 2*np.pi + atan2(pts[3,1] - SPV[7], pts[3,0] - SPV[6])
    ShapeInfo[2, 3, 1] = atan2(pts[3,1] - SPV[7], pts[3,0] - SPV[6]) if atan2(pts[3,1] - SPV[7], pts[3,0] - SPV[6]) > 0 \
                          else 2*np.pi + atan2(pts[3,1] - SPV[7], pts[3,0] - SPV[6])
    #-----------------------------------------------------------------------------------------------------------------------

    #--------------------------------------TILE 3---------------------------------------------------------------------------
    ShapeInfo[3, 0, 0] = dist([pts[0,0]-SPV[9], pts[0,1]-SPV[10]]) 
    ShapeInfo[3, 1, 0] = dist([pts[0,0]-SPV[9], pts[0,1]-SPV[10]])
    ShapeInfo[3, 2, 0] = dist([pts[7,0]-SPV[9], pts[7,1]-SPV[10]])
    ShapeInfo[3, 3, 0] = dist([pts[8,0]-SPV[9], pts[8,1]-SPV[10]])
    
    ShapeInfo[3, 0, 1] = atan2(pts[0,1] - SPV[10], pts[0,0] - SPV[9]) if atan2(pts[0,1] - SPV[10], pts[0,0] - SPV[9]) > 0 \
                          else 2*np.pi + atan2(pts[0,1] - SPV[10], pts[0,0] - SPV[9])
    ShapeInfo[3, 1, 1] = atan2(pts[0,1] - SPV[10], pts[0,0] - SPV[9]) if atan2(pts[0,1] - SPV[10], pts[0,0] - SPV[9]) > 0 \
                          else 2*np.pi + atan2(pts[0,1] - SPV[10], pts[0,0] - SPV[9])
    ShapeInfo[3, 2, 1] = atan2(pts[7,1] - SPV[10], pts[7,0] - SPV[9]) if atan2(pts[7,1] - SPV[10], pts[7,0] - SPV[9]) > 0 \
                          else 2*np.pi + atan2(pts[7,1] - SPV[10], pts[7,0] - SPV[9])
    ShapeInfo[3, 3, 1] = atan2(pts[8,1] - SPV[10], pts[8,0] - SPV[9]) if atan2(pts[8,1] - SPV[10], pts[8,0] - SPV[9]) > 0 \
                          else 2*np.pi + atan2(pts[8,1] - SPV[10], pts[8,0] - SPV[9])    
    #-----------------------------------------------------------------------------------------------------------------------                      
    
    #--------------------------------------TILE 4---------------------------------------------------------------------------
    ShapeInfo[4, 0, 0] = dist([pts[9,0]-SPV[12], pts[9,1]-SPV[13]]) 
    ShapeInfo[4, 1, 0] = dist([pts[8,0]-SPV[12], pts[8,1]-SPV[13]])
    ShapeInfo[4, 2, 0] = dist([pts[11,0]-SPV[12], pts[11,1]-SPV[13]])
    ShapeInfo[4, 3, 0] = dist([pts[10,0]-SPV[12], pts[10,1]-SPV[13]])
    
    ShapeInfo[4, 0, 1] = atan2(pts[9,1] - SPV[13], pts[9,0] - SPV[12]) if atan2(pts[9,1] - SPV[13], pts[9,0] - SPV[12]) > 0 \
                          else 2*np.pi + atan2(pts[9,1] - SPV[13], pts[9,0] - SPV[12])
    ShapeInfo[4, 1, 1] = atan2(pts[8,1] - SPV[13], pts[8,0] - SPV[12]) if atan2(pts[8,1] - SPV[13], pts[8,0] - SPV[12]) > 0 \
                          else 2*np.pi + atan2(pts[8,1] - SPV[13], pts[8,0] - SPV[12])
    ShapeInfo[4, 2, 1] = atan2(pts[11,1] - SPV[13], pts[11,0] - SPV[12]) if atan2(pts[11,1] - SPV[13], pts[11,0] - SPV[12]) > 0 \
                          else 2*np.pi + atan2(pts[11,1] - SPV[13], pts[11,0] - SPV[12])
    ShapeInfo[4, 3, 1] = atan2(pts[10,1] - SPV[13], pts[10,0] - SPV[12]) if atan2(pts[10,1] - SPV[13], pts[10,0] - SPV[12]) > 0 \
                          else 2*np.pi + atan2(pts[10,1] - SPV[13], pts[10,0] - SPV[12])    
    #-----------------------------------------------------------------------------------------------------------------------                          
    

    #--------------------------------------TILE 5---------------------------------------------------------------------------
    ShapeInfo[5, 0, 0] = dist([pts[3,0]-SPV[15], pts[3,1]-SPV[16]]) 
    ShapeInfo[5, 1, 0] = dist([pts[10,0]-SPV[15], pts[10,1]-SPV[16]])
    ShapeInfo[5, 2, 0] = dist([pts[4,0]-SPV[15], pts[4,1]-SPV[16]])
    ShapeInfo[5, 3, 0] = dist([pts[3,0]-SPV[15], pts[3,1]-SPV[16]])
    
    ShapeInfo[5, 0, 1] = atan2(pts[3,1] - SPV[16], pts[3,0] - SPV[15]) if atan2(pts[3,1] - SPV[16], pts[3,0] - SPV[15]) > 0 \
                          else 2*np.pi + atan2(pts[3,1] - SPV[16], pts[3,0] - SPV[15])
    ShapeInfo[5, 1, 1] = atan2(pts[10,1] - SPV[16], pts[10,0] - SPV[15]) if atan2(pts[10,1] - SPV[16], pts[10,0] - SPV[15]) > 0 \
                          else 2*np.pi + atan2(pts[10,1] - SPV[16], pts[10,0] - SPV[15])
    ShapeInfo[5, 2, 1] = atan2(pts[4,1] - SPV[16], pts[4,0] - SPV[15]) if atan2(pts[4,1] - SPV[16], pts[4,0] - SPV[15]) > 0 \
                          else 2*np.pi + atan2(pts[4,1] - SPV[16], pts[4,0] - SPV[15])
    ShapeInfo[5, 3, 1] = atan2(pts[3,1] - SPV[16], pts[3,0] - SPV[15]) if atan2(pts[3,1] - SPV[16], pts[3,0] - SPV[15]) > 0 \
                          else 2*np.pi + atan2(pts[3,1] - SPV[16], pts[3,0] - SPV[15])    
    #-----------------------------------------------------------------------------------------------------------------------                                                
      

    #--------------------------------------TILE 6---------------------------------------------------------------------------
    ShapeInfo[6, 0, 0] = dist([pts[7,0]-SPV[18], pts[7,1]-SPV[19]]) 
    ShapeInfo[6, 1, 0] = dist([pts[7,0]-SPV[18], pts[7,1]-SPV[19]])
    ShapeInfo[6, 2, 0] = dist([pts[6,0]-SPV[18], pts[6,1]-SPV[19]])
    ShapeInfo[6, 3, 0] = dist([pts[6,0]-SPV[18], pts[6,1]-SPV[19]])
    
    ShapeInfo[6, 0, 1] = atan2(pts[7,1] - SPV[19], pts[7,0] - SPV[18]) if atan2(pts[7,1] - SPV[19], pts[7,0] - SPV[18]) > 0 \
                          else 2*np.pi + atan2(pts[7,1] - SPV[19], pts[7,0] - SPV[18])
    ShapeInfo[6, 1, 1] = atan2(pts[7,1] - SPV[19], pts[7,0] - SPV[18]) if atan2(pts[7,1] - SPV[19], pts[7,0] - SPV[18]) > 0 \
                          else 2*np.pi + atan2(pts[7,1] - SPV[19], pts[7,0] - SPV[18])
    ShapeInfo[6, 2, 1] = atan2(pts[6,1] - SPV[19], pts[6,0] - SPV[18]) if atan2(pts[6,1] - SPV[19], pts[6,0] - SPV[18]) > 0 \
                          else 2*np.pi + atan2(pts[6,1] - SPV[19], pts[6,0] - SPV[18])
    ShapeInfo[6, 3, 1] = atan2(pts[6,1] - SPV[19], pts[6,0] - SPV[18]) if atan2(pts[6,1] - SPV[19], pts[6,0] - SPV[18]) > 0 \
                          else 2*np.pi + atan2(pts[6,1] - SPV[19], pts[6,0] - SPV[18])    
    #-----------------------------------------------------------------------------------------------------------------------                                                
                          
    #--------------------------------------TILE 7---------------------------------------------------------------------------
    ShapeInfo[7, 0, 0] = dist([pts[11,0]-SPV[21], pts[11,1]-SPV[22]]) 
    ShapeInfo[7, 1, 0] = dist([pts[6,0]-SPV[21], pts[6,1]-SPV[22]])
    ShapeInfo[7, 2, 0] = dist([pts[6,0]-SPV[21], pts[6,1]-SPV[22]])
    ShapeInfo[7, 3, 0] = dist([pts[5,0]-SPV[21], pts[5,1]-SPV[22]])
    
    ShapeInfo[7, 0, 1] = atan2(pts[11,1] - SPV[22], pts[11,0] - SPV[21]) if atan2(pts[11,1] - SPV[22], pts[11,0] - SPV[21]) > 0 \
                          else 2*np.pi + atan2(pts[11,1] - SPV[22], pts[11,0] - SPV[21])
    ShapeInfo[7, 1, 1] = atan2(pts[6,1] - SPV[22], pts[6,0] - SPV[21]) if atan2(pts[6,1] - SPV[22], pts[6,0] - SPV[21]) > 0 \
                          else 2*np.pi + atan2(pts[6,1] - SPV[22], pts[6,0] - SPV[21])
    ShapeInfo[7, 2, 1] = atan2(pts[6,1] - SPV[22], pts[6,0] - SPV[21]) if atan2(pts[6,1] - SPV[22], pts[6,0] - SPV[21]) > 0 \
                          else 2*np.pi + atan2(pts[6,1] - SPV[22], pts[6,0] - SPV[21])
    ShapeInfo[7, 3, 1] = atan2(pts[5,1] - SPV[22], pts[5,0] - SPV[21]) if atan2(pts[5,1] - SPV[22], pts[5,0] - SPV[21]) > 0 \
                          else 2*np.pi + atan2(pts[5,1] - SPV[22], pts[5,0] - SPV[21])    
    #-----------------------------------------------------------------------------------------------------------------------                                                
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                         
                          
                          
                          
                          
    #---------------------------------------ENCODE THE CONNECTION AMONG THE PARTICLES---------------------------------------
    '''ConnectedPolygonsMatrix(CPM) tells which two bodies are connected. The connections are first considered 'row- wise' 
    and than 'column-wise'. Every row of CPM denotes a connection.'''            
    CPM = np.zeros((numberOfConnections,2), dtype = int); j = 0
                                                                                                          
    for RowCxnNum in range(n):                                                                                                                
        for i in range (1 , n): 
            CPM[j] = [i + n*RowCxnNum-1, i + n*RowCxnNum + 1-1]
            j = j + 1
            
    for ColumnCxnNum in range(n):                                                                                                                
        for i in range (1 , n):
            CPM[j] = [ColumnCxnNum + n*i-(n-1)-1, ColumnCxnNum + n*i-(n-1) + n-1]
            j = j + 1        
    #-----------------------------------------------------------------------------------------------------------------------
    CPM = np.delete(CPM, [5,11], 0)
    
    
    #--------------------------------------2D AND 3D BOOLEAN CONNECTED POLYGONS MATRIX--------------------------------------
    '''The connection among the polygons is represented in Boolean fashion. The number of rows in BooleanConnectedPolygonsM-
    trix2D(BCPM2D) = numberOfConnections. The number of column in the BooleanCPM is equal to len(PositionVector)'''
    BCPM2D =  np.zeros((numberOfConnections-2, len(SPV)), dtype = int)
    BCPM3D =  np.zeros((numberOfConnections-2, len(SPV), len(SPV)), dtype = int)
    
    for i in range(numberOfConnections-2):
        BCPM2D[i, 3*CPM[i,0]:3*CPM[i,0]+3] = 1  
        BCPM2D[i, 3*CPM[i,1]:3*CPM[i,1]+3] = 1
    
    for i in range(numberOfConnections-2):
        BCPM3D[i, 3*CPM[i,0]:3*CPM[i,0]+3, 3*CPM[i,0]:3*CPM[i,0]+3] = 1  
        BCPM3D[i, 3*CPM[i,1]:3*CPM[i,1]+3, 3*CPM[i,1]:3*CPM[i,1]+3] = 1 
        BCPM3D[i, 3*CPM[i,0]:3*CPM[i,0]+3, 3*CPM[i,1]:3*CPM[i,1]+3] = 1
        BCPM3D[i, 3*CPM[i,1]:3*CPM[i,1]+3, 3*CPM[i,0]:3*CPM[i,0]+3] = 1
        
    #-----------------------------------------------------------------------------------------------------------------------                                   
                                       
    
    
    #--------------------------------ENCODE WHICH CORNERS ARE CONNECTED IN A CONNECTION-------------------------------------
    '''A row of ConnectedCornersMatrix(CCM) tells the Corner Number of the Polygon that share the Connection'''
    CCM = np.zeros((numberOfConnections,2), dtype = int)        
    CCM[0:int(numberOfConnections/2)]=[3,1]
    CCM[int(numberOfConnections/2):numberOfConnections]=[2,0]
    #-----------------------------------------------------------------------------------------------------------------------
    CCM = np.delete(CCM, [5,11], 0)
    
    
    
    
    
    
    
    
    
    theta  = conc((lnsp(0, 1.04, 10).reshape(10,1),lnsp(0, -1.04, 10).reshape(10,1)))
    #theta  = lnsp(0, -1.04, 10).reshape(10,1)    
    #theta = np.array([0.0])
    #theta = np.array([-1.04])    
    #theta =  np.array([0.0, np.pi/6, np.pi/3])#, 0.0, -np.pi/6, -np.pi/3])
    thetaCount     = 0
    FinalEnergy    = np.zeros([len(theta), 1])
    Distance       = np.zeros([len(theta), 1])
    PositionVector = SPV      
    
    #FigNumber = 0; #plt.figure(FigNumber); #FigNumber += 1
    #plt.figure(1)
    #visualize(PositionVector, n, ShapeInfo)
    

    for PositionVector[5] in theta:
        
        PositionVector = SPV if PositionVector[5] == 0.0 else PositionVector
            
        #---------------------------------X1, Y1, Theta1, L1, Phi1, X2, Y2, Theta2, L2, Phi2--------------------------------
        ''' The 10 required vectors for Vectorization'''
        X1 = np.zeros((numberOfConnections -2,));Y1 = np.copy(X1);Theta1 = np.copy(X1);LenVec1 =np.copy(X1);Phi1=np.copy(X1)
        X2 = np.zeros((numberOfConnections -2,));Y2 = np.copy(X1);Theta2 = np.copy(X1);LenVec2 =np.copy(X1);Phi2=np.copy(X1)
        
        for i in range(numberOfConnections-2):
            X1[i]       = PositionVector[3*CPM[i, 0], 0]   
            Y1[i]       = PositionVector[3*CPM[i, 0] + 1, 0]
            Theta1[i]   = PositionVector[3*CPM[i, 0] + 2, 0]
            LenVec1[i]  = ShapeInfo[CPM[i, 0], CCM[i, 0],0]
            Phi1[i]     = ShapeInfo[CPM[i, 0], CCM[i, 0],1]
            
            X2[i]       = PositionVector[3*CPM[i, 1], 0]
            Y2[i]       = PositionVector[3*CPM[i, 1] + 1, 0]
            Theta2[i]   = PositionVector[3*CPM[i, 1] + 2, 0]
            LenVec2[i]  = ShapeInfo[CPM[i, 1], CCM[i, 1],0]
            Phi2[i]     = ShapeInfo[CPM[i, 1], CCM[i, 1],1]
        #-------------------------------------------------------------------------------------------------------------------
            
           
        #----------------------NON LINEAR CONJUGATE GRADIENT ALGORITHM WITH NEWTON RAPHSON AND FLETCHER REEVS---------------
        '''Conjugate gradient algorithm to search for the equilibrium state of the mechanism for a given configuration''' 
        
        j = 0 ; j_max = 5000 ; k = 0 
        
        Network = UpdateNetwork(X1, Y1, Theta1, LenVec1, Phi1, X2, Y2, Theta2, LenVec2, Phi2, BCPM2D, BCPM3D)
        FirstDerivativeVector  = Network.UpdateFirstDerivativeVector()
        SecondDerivativeMatrix = Network.UpdateSecondDerivativeMatrix()
        EnergyInSprings        = np.zeros([j_max,1])
        Iterations             = np.zeros([j_max,1])
        EnergyInSprings[j, 0]  = Network.UpdateEnergyInSprings()
        Iterations[j,0]        = j   
        
        
        r       =  -FirstDerivativeVector
        d       =  r
        del_new =  dotProduct(transpose(r), r)
        #del_o   =  del_new
        
        while (j < j_max):
            
            #del_d = dotProduct(transpose(d), d)
            alpha = -(dotProduct(transpose(FirstDerivativeVector), d))/  \
                     (dotProduct(transpose(d),dotProduct(SecondDerivativeMatrix,d)))
            
            PositionVector = PositionVector + alpha*d
            
            #-------------------------------X1, Y1, Theta1, L1, Phi1, X2, Y2, Theta2, L2, Phi2------------------------------
            ''' The 10 required vectors for Vectorization'''        
            for i in range(numberOfConnections-2):
                X1[i]     = PositionVector[3*CPM[i, 0], 0]   
                Y1[i]     = PositionVector[3*CPM[i, 0] + 1, 0]
                Theta1[i] = PositionVector[3*CPM[i, 0] + 2, 0]
                LenVec1[i]= ShapeInfo[CPM[i, 0], CCM[i, 0],0]
                Phi1[i]   = ShapeInfo[CPM[i, 0], CCM[i, 0],1]
                
                X2[i]     = PositionVector[3*CPM[i, 1], 0]
                Y2[i]     = PositionVector[3*CPM[i, 1] + 1, 0]
                Theta2[i] = PositionVector[3*CPM[i, 1] + 2, 0]
                LenVec2[i]= ShapeInfo[CPM[i, 1], CCM[i, 1],0]
                Phi2[i]   = ShapeInfo[CPM[i, 1], CCM[i, 1],1]
            #---------------------------------------------------------------------------------------------------------------
                
            
            Network = UpdateNetwork(X1, Y1, Theta1, LenVec1, Phi1, X2, Y2, Theta2, LenVec2, Phi2, BCPM2D, BCPM3D)
            FirstDerivativeVector      = Network.UpdateFirstDerivativeVector()
            SecondDerivativeMatrix     = Network.UpdateSecondDerivativeMatrix()
            
            if(j < j_max - 1):
                EnergyInSprings[j + 1, 0]  = Network.UpdateEnergyInSprings()
                Iterations[j + 1,0]        = j + 1
        
            
            
            r       = -FirstDerivativeVector
            del_old = del_new
            del_new = dotProduct(transpose(r), r)
            beta    = del_new/del_old
            d       = r + beta*d
    
            
            #------------------LOOP RESTART------------------------
            k = k + 1
            if (k==24) or (dotProduct(transpose(r), d) <= 0):
                d = r
                k = 0
            #------------------------------------------------------
              
              
            #------------BREAK OUT IF CONVERGENCE------------------ 
            E1  =  log10(EnergyInSprings[EnergyInSprings != 0][-1])  
            E2  =  log10(EnergyInSprings[EnergyInSprings != 0][-2])
            if (  E1   < -10 or E2 - E1 < 0.00001):
                break
            #------------------------------------------------------
            j = j + 1  
             
        #-------------------------------------------------------------------------------------------------------------------      
         
        '''
        #--------CONVERGENCE CURVE AND FINAL TILING---------  
        plt.figure(FigNumber); FigNumber += 1
        plt.plot(Iterations, np.log10(EnergyInSprings), '.')
        plt.xlabel('Iterations'); plt.ylabel('Energy')
        '''
        
        #plt.figure(FigNumber) ; FigNumber += 1
        #visualize(PositionVector, n, ShapeInfo)
        #visualize(PositionVector, n, ShapeInfo)
        #---------------------------------------------------
        #print(PositionVector)
               
        
        #print(len(EnergyInSpri))
        #--------CONVERGENCE CURVE AND FINAL TILING---------  
        #plt.figure(FigNumber); FigNumber += 1
        #plt.figure(1)
        #plt.plot(Iterations[0:j], EnergyInSprings[0:j])
        #plt.xlabel('Iterations'); plt.ylabel('Energy')
        
        
        #plt.figure(FigNumber); FigNumber += 1
        #visualize(PositionVector, n, ShapeInfo)
            #--------------------------------------------------- 
        
        '''          
        plt.plot((ShapeInfo[0,0,0]*np.cos(ShapeInfo[0,0,1]+PositionVector[2])+PositionVector[0]), \
                (ShapeInfo[0,0,0]*np.sin(ShapeInfo[0,0,1]+PositionVector[2])+PositionVector[1]), 'o', color = 'red')
        plt.plot((ShapeInfo[0,1,0]*np.cos(ShapeInfo[0,1,1]+PositionVector[2])+PositionVector[0]), \
                (ShapeInfo[0,1,0]*np.sin(ShapeInfo[0,1,1]+PositionVector[2])+PositionVector[1]), 'o', color = 'blue')
                
        plt.plot((ShapeInfo[2,3,0]*np.cos(ShapeInfo[2,3,1]+PositionVector[8])+PositionVector[6]), \
                (ShapeInfo[2,3,0]*np.sin(ShapeInfo[2,3,1]+PositionVector[8])+PositionVector[7]), 'o', color = 'blue')
        plt.plot((ShapeInfo[2,0,0]*np.cos(ShapeInfo[2,0,1]+PositionVector[8])+PositionVector[6]), \
                (ShapeInfo[2,0,0]*np.sin(ShapeInfo[2,0,1]+PositionVector[8])+PositionVector[7]), 'o', color = 'red')
        
        plt.plot((ShapeInfo[6,1,0]*np.cos(ShapeInfo[6,1,1]+PositionVector[20])+PositionVector[18]), \
                (ShapeInfo[6,1,0]*np.sin(ShapeInfo[6,1,1]+PositionVector[20])+PositionVector[19]), 'o', color = 'blue')
        plt.plot((ShapeInfo[6,2,0]*np.cos(ShapeInfo[6,2,1]+PositionVector[20])+PositionVector[18]), \
                (ShapeInfo[6,2,0]*np.sin(ShapeInfo[6,2,1]+PositionVector[20])+PositionVector[19]), 'o', color = 'red')
               
        '''
        #----------------------------------------FINAL INFORMATION OF THE TILING--------------------------------------------
        FinalEnergy[thetaCount, 0] = E1
        Distance[thetaCount, 0] = sqrt(((ShapeInfo[5,2,0]*cos(ShapeInfo[5,2,1]+PositionVector[17,0])+PositionVector[15,0])-\
                (ShapeInfo[7,3,0]*cos(ShapeInfo[7,3,1]+PositionVector[23,0])+PositionVector[21,0]))**2+((ShapeInfo[5,2,0]* \
                sin(ShapeInfo[5,2,1]+PositionVector[17,0])+ PositionVector[16,0])-(ShapeInfo[7,3,0]*sin(ShapeInfo[7,3,1]+  \
                PositionVector[23,0])+PositionVector[22,0]))**2)
        thetaCount += 1               
        #------------------------------------------------------------------------------------------------------------------- 
    #FigNumber += 1    
    Distance[0:10] = np.flipud(Distance[0:10])    
    theta[0:10] = np.flipud(theta[0:10])    
    
    '''
    #-----------------------------------------------DISTANCE CURVE----------------------------------------------------------
    #plt.figure(FigNumber); FigNumber += 1
    plt.plot(theta*180/np.pi, Distance, color='blue')
    plt.ylim((0.5,1.5)); plt.xlim((-75,75))
    plt.ylabel(r'D',fontsize=35,fontname = "Times New Roman", labelpad=30, rotation = 0)
    plt.xlabel(r'$\mathrm{\theta^{\circ}}$',fontsize=30, fontname = "Times New Roman", labelpad = 4)
    #plt.xticks([-60, 0, 60], fontsize = 18); plt.yticks([8.0, 6.0,  4.0, 2.0, 0.0], fontsize = 20)    

    #-----------------------------------------------------------------------------------------------------------------------                    
    '''
    '''    
    
                                                  
    #-----------------------------------------------EQUILIBRATED ENERGY CURVE-----------------------------------------------
    plt.figure(FigNumber); marker = ['o','s','d','D','*']; color = ['black','red','green','blue','orange']
    plt.plot(theta*180/np.pi, FinalEnergy, 'o', color='blue') 
    plt.ylim((-18,0))#; FigNumber += 1
    plt.xlabel(r'$\mathrm{\theta^{\circ}}$',fontsize=20,labelpad=0)
    plt.ylabel('$log_{10}(E_{Final})$',fontsize=20,rotation = 90)
    #-----------------------------------------------------------------------------------------------------------------------    
    '''
     
    yTarget = 1.0
    #objVal = sum(abs(Distance - yTarget), 0)
    #print (FinalEnergy)
    #print(Distance)
    

    return np.sqrt(sum(abs(Distance - yTarget)**2, 0)/20)
            
    #return EnergyInSprings
    






