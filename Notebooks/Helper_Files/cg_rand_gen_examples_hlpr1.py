# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:42:19 2016

@author: singh
"""



# ----------------------------------------------------IMPORT ALL THE PACKAGES-------------------------------------------
import numpy as np; import matplotlib;from matplotlib import pyplot as plt;import pylab;from scipy.spatial import distance       
import matplotlib.patches as patches;import random; import scipy; import math; import sympy as sp; from sympy import  * 
from scipy.spatial  import distance ; import time; start_time = time.time()
import os
# ----------------------------------------------------------------------------------------------------------------------


#----------------------------------------------THESE NOTATIONS MAKE THE WRITIGN EASIER----------------------------------
pi = np.pi ; dot = np.dot ;  sin = np.sin ;  cos = np.cos ;  ar = np.array ;  sqrt = np.sqrt; rand = scipy.rand  
arange = scipy.arange; show = pylab.show; plot = pylab.plot; axis = pylab.axis; grid = pylab.grid; title  = pylab.title 
atan = np.arctan; transpose = np.transpose ; dotProduct = np.dot
#-----------------------------------------------------------------------------------------------------------------------


plt.close('all')


#--------------------------------------------------------------OOP PART-------------------------------------------------
class UpdateNetwork(object):
    
    
    def __init__(self, X1, Y1, Theta1, LenVec1, Phi1, X2, Y2, Theta2, LenVec2, Phi2, BCPM2D, BCPM3D): 
        """Initialize the network by defining the current position vector, the connectivity among the members, the shape
        information of the members. Preallocate the First Derivative Vector(FDV) and the  Second  Derivative Matrix(SDM) 
        for the netwrok."""
        
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
        
        self.FDV[2,0] = 0.0
        self.FDV[5,0] = 0.0 
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
        
        self.SDM[:,5,5]       +=  2*LenVec2*(LenVec2*sin(Theta2 + Phi2)**2 + LenVec2*cos(Theta2 + Phi2)**2 + (X1 - X2 +\
                                  LenVec1*cos(Theta1 + Phi1)- LenVec2*cos(Theta2 + Phi2))*cos(Theta2 + Phi2) + (Y1 - Y2\
                                  + LenVec1*sin(Theta1 + Phi1) -LenVec2*sin(Theta2 + Phi2))*sin(Theta2 + Phi2))
                                  
        self.SDM = self.SDM.flatten()[(np.cumsum(self.BCPM3D).reshape(self.BCPM3D.shape)-1)*self.BCPM3D] * self.BCPM3D  
        self.SDM = np.sum(self.SDM, axis = 0)
        
        self.SDM[2, :] = self.SDM[:, 2] = 0.0
        self.SDM[5, :] = self.SDM[:, 5] = 0.0        
        return self.SDM


    def EnergyStoredInSprings(self):
        X1 = self.X1; Y1 = self.Y1; Theta1 = self.Theta1; LenVec1 = self.LenVec1; Phi1 = self.Phi1  
        X2 = self.X2; Y2 = self.Y2; Theta2 = self.Theta2; LenVec2 = self.LenVec2; Phi2 = self.Phi2 
        self.Energy = ((LenVec1*np.cos(Phi1+Theta1)+X1)-(LenVec2*np.cos(Phi2+Theta2)+X2))**2+((LenVec1*np.sin(Phi1+\
                        Theta1)+Y1)-(LenVec2*np.sin(Phi2+Theta2)+Y2))**2
        self.Energy = np.sum(self.Energy, axis = 0)                
        return self.Energy                        
#-----------------------------------------------------------------------------------------------------------------------
         



def my_func():        
    n = 3 ; numberOfPolygons = n**2
    numberOfConnections =  int((2*4  +  3*4*(n-2)  +  4*(n**2 - 4*(n-2) -4))/2) 



    #--------------------------------------DEFINE THE POSITION VECTOR OF THE PARTICLES--------------------------------------
    '''startingPositionVector encodes the position and orientation of the polygons. Preallocate the position vector. Fill in 
    the appropriate places of the vector and randomize things a bit.'''                                
    startingPositionVector = np.zeros((3*(numberOfPolygons - 1), 1), dtype = float)   

    for Polygon in range(numberOfPolygons - 1):
        startingPositionVector[3*Polygon : 3*Polygon+3] = np.array([[10+10*(Polygon%n)], [10-10*(Polygon//n)], [0]])                           
    #-----------------------------------------------------------------------------------------------------------------------




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
    BCPM2D =  np.zeros((numberOfConnections-2, len(startingPositionVector)), dtype = int)
    BCPM3D =  np.zeros((numberOfConnections-2, len(startingPositionVector), len(startingPositionVector)), dtype = int)

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


    #------------------------------------ENCODE THE SHAPE INFO INTO ALL THE PARTICLES---------------------------------------
    #Define all the 'l' vectors (LVectors) originating from the center of the particles and define all the 'theta'(LAngl-
    #es) of each of the 'l' vectors

    LVectors = np.zeros((n**2, 4), dtype = float) 
    LAngles  = np.zeros((n**2, 4), dtype = float)        
    for Polygon in range(numberOfPolygons - 1):        
        #LVectors[Polygon] = [5.0, 6.0, 7.0, 5]
        LVectors[Polygon] = [1.0, 1.0, 1.0, 1.0]
        LAngles[Polygon]  = [np.pi/2, 2*np.pi/2, 3*np.pi/2, 4*np.pi/2]                                          
        #LAngles[Polygon]  = [np.pi/3, np.pi, 2*np.pi/3+np.pi, np.pi/3+2*np.pi/3+np.pi]
    #Fill the Shape Info in a np.array with shape (n**2, 4, 2) and randomize things a bit    
    ShapeInfo = np.zeros((numberOfPolygons - 1, 4, 2), dtype = float)       
    for Polygon in range(numberOfPolygons - 1):
        ShapeInfo[Polygon, :, 0]  = [L +  1.0*(0.50 - random.random()) for L in LVectors[Polygon]]                         
        ShapeInfo[Polygon, :, 1]  = [Theta + 1.0*(0.50 - random.random()) for Theta in LAngles[Polygon]]                

    #-----------------------------------------------------------------------------------------------------------------------





    equilibratedEnergy = [];  distance = []

    OrientationBody2 = np.concatenate((np.linspace(0, 1.04, 10).reshape(10,1),np.linspace(0, -1.04, 10).reshape(10,1)))
    OrientationBody2 = np.array([0])
    PositionVector = startingPositionVector  
    FigNumber = 0

    #plt.figure(FigNumber); FigNumber += 1
    #visualize(PositionVector, n, ShapeInfo)

    AngleCount = 0

    for PositionVector[5] in OrientationBody2:

        EnergyOfSystem = []; iterCount = []
        AngleCount += 1
        if(PositionVector[5] == 0.0):
            PositionVector = startingPositionVector

        #---------------------------------X1, Y1, Theta1, L1, Phi1, X2, Y2, Theta2, L2, Phi2------------------------------------
        ''' The 10 required vectors for Vectorization'''
        X1 = np.zeros((numberOfConnections -2,)); Y1 = np.copy(X1); Theta1 = np.copy(X1); LenVec1 = np.copy(X1); Phi1 = np.copy(X1)
        X2 = np.zeros((numberOfConnections -2,)); Y2 = np.copy(X1); Theta2 = np.copy(X1); LenVec2 = np.copy(X1); Phi2 = np.copy(X1)

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
        #-----------------------------------------------------------------------------------------------------------------------


        #----------------------NON LINEAR CONJUGATE GRADIENT ALGORITHM WITH NEWTON RAPHSON AND FLETCHER REEVS-----------

        Network = UpdateNetwork(X1, Y1, Theta1, LenVec1, Phi1, X2, Y2, Theta2, LenVec2, Phi2, BCPM2D, BCPM3D)
        FirstDerivativeVector  = Network.UpdateFirstDerivativeVector()
        SecondDerivativeMatrix = Network.UpdateSecondDerivativeMatrix()
        EnergyOfSystem.append(Network.EnergyStoredInSprings()); count = 0; iterCount.append(count)

        j = 0 ; j_max = 1000 ; k = 0 

        r =  -FirstDerivativeVector

        d = r

        del_new = dotProduct(transpose(r), r)
        del_o = del_new


        while (j < j_max):

            del_d = dotProduct(transpose(d), d)


            alpha = -(dotProduct(transpose(FirstDerivativeVector), d))/  \
            (dotProduct(transpose(d),dotProduct(SecondDerivativeMatrix,d)))


            PositionVector = PositionVector + alpha*d


            #-----------------------------------------------------------------------------------------------------------
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
            #-----------------------------------------------------------------------------------------------------------


            Network = UpdateNetwork(X1, Y1, Theta1, LenVec1, Phi1, X2, Y2, Theta2, LenVec2, Phi2, BCPM2D, BCPM3D)
            FirstDerivativeVector = Network.UpdateFirstDerivativeVector()
            SecondDerivativeMatrix = Network.UpdateSecondDerivativeMatrix()
            EnergyOfSystem.append(Network.EnergyStoredInSprings()); count += 1; iterCount.append(count)



            r =  -FirstDerivativeVector

            del_old = del_new
            del_new = dotProduct(transpose(r), r)

            beta = del_new/del_old
            d = r + beta*d

            #------------------LOOP RESTART------------------------
            k = k + 1
            if (k==27) or (dotProduct(transpose(r), d) <= 0):
                d = r
                k = 0
            #------------------------------------------------------

            E1 = np.log10(EnergyOfSystem[-1])
            E2 = np.log10(EnergyOfSystem[-2])
            #if (np.log10(EnergyOfSystem[-1]) < -20):
            if(E2 - E1 < 0.00001):
                #print('{0}/{1}'.format(AngleCount,len(OrientationBody2)))
                break

            j = j + 1  

        #---------------------------------------------------------------------------------------------------------------      
        
        equilibratedEnergy.append(EnergyOfSystem[-1])

    return(equilibratedEnergy[-1])
        



                     


    p











   