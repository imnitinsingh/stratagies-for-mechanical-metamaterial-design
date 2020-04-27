import numpy as np
dist = np.linalg.norm
atan = np.arctan
numberOfPolygons = 9
from math import atan2


def hlpr_fun(pts):
    SPV = np.zeros((3*(numberOfPolygons - 1), 1), dtype = float)   
                                             
    SPV[0],  SPV[1]  = (pts[0]+pts[1])/2;    SPV[2] = 0
    SPV[3],  SPV[4]  = (pts[1]+pts[2]+pts[9])/3;  SPV[5] = 0                            
    SPV[6],  SPV[7]  = (pts[2]+pts[3])/2;  SPV[8] = 0
    SPV[9],  SPV[10] = (pts[0]+pts[7]+pts[8])/3; SPV[11] = 0    
    SPV[12], SPV[13] = (pts[8]+pts[9]+pts[10]+pts[11])/4; SPV[14] = 0    
    SPV[15], SPV[16] = (pts[3]+pts[4]+pts[10])/3; SPV[17] = 0 
    SPV[18], SPV[19] = (pts[6]+pts[7])/2; SPV[20] = 0
    SPV[21], SPV[22] = (pts[5]+pts[6]+pts[11])/3;  SPV[23] = 0
	
	
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
	
    return(SPV, ShapeInfo) 	