# -*- coding: utf-8 -*-
import numpy as np
import math
import numba
from numba import vectorize

#General introduction

#When evaluating images of semi-transparent samples, the origin of the light
#is not completely confined to the focal plane of the objective lens of the camera/microscope.
#Therefore one can simulate the optics inside the probe up to the depth,
#from which on one expects no more substantial contribution. 

#The sample and a terminating plane, corresponding to the focal plane of the camera,
#can be given as a rectangular grid, representing the geometry of the problem.

#An initial distribution of light rays propagate through the grid until all rays
#have either passed the focal plane, decayed to a certain threshold or left the simulation
#crossing the boundaries.
#In every loop all rays reach the interface of two grid cells and propagate
#by splitting into two rays. One travelling into the neighbouring cell via refraction
#and one being reflected staying the same cell. 

#Code description

#The variable i represents the rays.
#It contains in 3 columns the x,y,z-components of the direction of the rays.
#Therefore it has for N rays the form of a Nx3-array.
#i_pos contains in the same way the origin of each ray.
#All variables containing n in their name, relate to surface normal vectors.
#n has again the form of a Nx3-array.
#Since n is already used, the refractive index is called eta.
#At the beginning the intensity of every ray is one.
#The variable intensity is threrefore one column of length N.

#The raytracer works on a rectangular 3-dimensional grid.
#Each grid cell has a defined size and the properties: refractive index and absorption coeff.
#The variable grid is a 4-dimensional matrix [xdim,ydim,zdim,info].
#The first 3 dimensions, containing each one entry with a positive integer,
#store the position of the cell relative to the other cells, like a 3-dimensional index.
#Setting these 3 integer one can choose for example the 000-cell or the 101-cell and so on. 
#The fourth dimension has 8 entries, in which the boundaries of the cell and the properties are stored,
#like: xlow,xhigh,ylow,yhigh,zlow,zhigh,eta,absorb
#grid[0,2,2,0] would give the lower boundary in x of the 022-cell

#The variables grid_index and grid_properties map the grid on to each light ray.
#Therefore grid_index stores for each ray the information in which cell the ray is.
#The cell is thereby specified by 3 indices, relating to the first 3 dimensions of the variable grid.
#So grid_index has the the form Nx3 with 3 columns for the indices and N rows for each ray.
#The variable grid_properties stores the 8 entries: xlow,xhigh,ylow,yhigh,zlow,zhigh,eta,absorb
#of the cell, containing the ray, for each light ray and has the form Nx8.
#Once grid_properties is constructed grid_index is not really needed any more.
#But to construct grid_properties again after a ray travels to a different cell grid_index is necessary.

#The cells forming the spatial boundary of the simulation, in which rays shall be neglected,
#are marked with a certain absorption coeff.=1000.
#The cells constituting the focal plane are marked with the absorption coeff.=-1000,
#it is assumed that the focal plane is a compact plane.
#If the refractive index of a cell is 0, the cell can not be penetrated by light
#and its surfaces are modelled with 100% reflectivity, corresponding to a metal surface.
 
#The objective of the camera or microscope can be specified by its numerical aperture.


#The simulation can be divided in three big functions and some small ones around them.
#The big three are: the simulation() itself, propagate() and interface_event().

#simulation()
#simulation() initializes and constructs all other variables
#and starts the loop of propagation.
#(the only variable really necessary is the grid,
#but of course a lot more variables can be given for specification)


#propagate()
#propagate() calculates intensity loss due to absorption,
#determines which interface will be hit by each ray,
#starts the function interface_event()
#and constructs with the result the image of the focal plane.

#interface_event()
#interface_event() starts the function interface_decision(), which determines
#which process happens at each interface depending on the refractive indices
#and the angle of incidence.(nothing if same refractive index on both sides,
#reflection & refraction, 100% reflection at metal, Total Internal Reflection)
#With the result of interface_decision the event is executed and
#all corresponding parameters are updated.
#Then the rays are split into two groups, one terminating at the focal plane
#and one staying in the simulation.

#For the functions propagate() and interface_event() nearly all variables are needed,
#so calling these functions looks a bit ugly, but is nothing complicated.

#Some functions start either with fast_ or gu_. These function have been optimized
#to a quasi-C-implementation to run faster.

def simulation(grid,
               number_of_rays=10**5,
               startvolume_borders=0,#would be set to the non-boundary grid volume
               chip_resolution=np.array([20,20]),
               focal_plane_size=0, #would take the y and z borders from startvolume_borders
               intensityborder=0.001,
               opening_angle_x=90,#assuming chip in yz-plane
               aperture=2,
               mask=False,
               startdistr=0,
               smallprintmode=True,
               printmode=False,
               mode='normal',#'test' and 'two_dim_xz' are possible
               history_num=-1,
               absorp_boundary=1000,
               absorp_final=-1000,
               focal_plane_direction=np.array([1,0,0]),
               ):
   
 
    #Configurations ###########################################
   
    #testmode configurations
    #in testmode the direction and position of the rays are not random and
    #therefore can be directly set
    #all rays have the same direction and position implying, one big ray
    if mode=='test':
        x_pos=350.
        y_pos=18.
        z_pos=280
       
        x_direc=0.5
        y_direc=0.4#0.001
        z_direc=0.4#.5
   
   
    #mask configurations
    #mask a certain volume, in which it is wanted to have no light sources
    #define the volume by its borders in x,y,z
    #if a second mask is needed, copy the following code and change _1 to _2
    #do the same at the "applying mask" point in the code
    if mask:
        xmasklow_1  =0
        xmaskhigh_1 =1200
        ymasklow_1  =175
        ymaskhigh_1 =200
        zmasklow_1  =98
        zmaskhigh_1 =100
   
    #End of configurattions ######################################
   
   
    #Initializations #############################################

    #Default initialization in case startvolume_borders or chip_size are not given
    if np.sum(startvolume_borders)==0:
        startvolume_borders=np.zeros(6)
        startvolume_borders[1]=grid[-2,0,0,1]
        startvolume_borders[3]=grid[0,-2,0,3]
        startvolume_borders[5]=grid[0,0,-2,5]
    if np.sum(focal_plane_size)==0:
        focal_plane_size=np.zeros(4)
        focal_plane_size[:2]=startvolume_borders[2:4]
        focal_plane_size[2:]=startvolume_borders[4:6]
    #End of default initializations

   
    #initializing positions
    if mode != 'test':
        i_pos=initialize_ray_positions(startdistr,startvolume_borders,number_of_rays)      
    else:
        i_pos=np.zeros([number_of_rays,3])
        i_pos[:,0]=x_pos
        i_pos[:,1]=y_pos
        i_pos[:,2]=z_pos
       
       
    #initializing directions
    if mode != 'test':
        i=initialize_ray_directions(opening_angle_x,number_of_rays,mode)
    else:
        i=np.zeros([number_of_rays,3])
        i[:,0]=x_direc
        i[:,1]=y_direc
        i[:,2]=z_direc

        i=normalize(i)
       
       
    #applying mask
    if mask:
        i_pos,i= mask_volume(i_pos,i,xmasklow_1,xmaskhigh_1,ymasklow_1,ymaskhigh_1,zmasklow_1,zmaskhigh_1)

   
    #initialize the intensity of every ray with one
    intensity=np.ones(len(i_pos[:,0]))   
   
    #with zeros initialized image of the camera chip   
    final=np.zeros(chip_resolution)   
   
    #variable which stores the history of a ray
    #for example number of reflections, refractions ...
    history=np.zeros(len(i_pos[:,0]))
   
    #yz-matrix which records the amount of intensity lost in each cell by absorption
    absorp_cross_section_yz=np.zeros(np.shape(grid)[0:3])
   
    #saves the information, in which grid cell each ray is contained
    grid_index=construct_grid_index(i_pos,grid)
   
    #saves the properties of the cell of each light ray
    grid_properties=construct_properties_from_index(grid_index,grid)
   
    #verbleib = number of rays still in the simulation
    verbleib=len(i_pos)
    counter=0
   
    DIM=len(grid[:,0,0])*len(grid[0,:,0])*len(grid[0,0,:])
   
    #only usefull in testmode, gives the box of each lightray in each loop
    index_protocoll=[]
   
    #gives general information about the simulation status in each loop
    #and can be shown during the simulation with printmode
    intensity_protocoll=[]
   
    #End of intializations ##############################################
   
    #Starting simulation ################################################
    #Running the simulation until the number of rays remaining is 0
    nochange=0
    while verbleib>0:

        old_grid_index=np.copy(grid_index)
        old_final=np.copy(final)
        #Running the function propagate(), which propagates each ray from its
        #current grid cell to the next one (or in case of reflection, in the same again).
        #It updates all rays and also constructs cumulatively in each loop
        #the image on the chip, with the variable final
        [i_pos,i,intensity,grid_index,grid_properties,
         final,index_protocoll,intensity_protocoll,history,
         absorp_change_out]   =   propagate(
                              i_pos,i,intensity,grid_index,
                              grid_properties,grid,final,focal_plane_direction,
                              focal_plane_size,intensityborder,index_protocoll,intensity_protocoll,
                              history,printmode,history_num,absorp_boundary,absorp_final,
                              chip_resolution,aperture,mode,smallprintmode)
       
        absorp_cross_section_yz=inside_absorp(grid,old_grid_index,absorp_change_out,absorp_cross_section_yz)       
        verbleib=len(i_pos)
        counter=counter+1
       
        if np.sum(final)==np.sum(old_final):
            nochange+=1
        else:
            nochange=0
       
        if nochange>DIM:
            #print "no change"
            break
       
       
        loop_printer(smallprintmode,printmode,verbleib,counter)
         
    #End of simulation #################################################         
    return final,index_protocoll,np.array(intensity_protocoll),absorp_cross_section_yz




def propagate(i_pos,i,intensity,grid_index,grid_properties,grid,final,focal_plane_direction,
              focal_plane_size,intensityborder,index_protocoll,intensity_protocoll,history,printmode,
              history_num,absorp_boundary,absorp_final,chip_resolution,aperture,mode,smallprintmode):
   
    #Calculating cell_n the vectors normal to the interfaces, which are hit by the rays
    #and the distance the rays travel to reach this interface with the function aim()
    cell_n,time_distance=aim(i,i_pos,grid_properties[:,0:2],grid_properties[:,2:4],grid_properties[:,4:6])
   
    #Updating position and cell index and cell properties
    #new is related to aim and could also be called neighbouring_cell
    new_grid_index=grid_index+cell_n       
    new_grid_properties=construct_properties_from_index(new_grid_index,grid)
   
    new_i_pos,new_intensity,absorp_change_out=fast_part_prop(i,i_pos,time_distance,grid_properties[:,7],intensity)

   
    #Running the function interface_event(), which returns two new arrays of updated light rays.
    #One array covers the rays staying in the simulation and one covers the rays terminating at the focal plane.
    #The first array can contain more rays than before, since reflection and refraction
    #can split one ray into two.
    [final_i_pos,final_i_refr,final_intensity,recurrent_i_pos,recurrent_i,recurrent_intensity,
     recurrent_grid_index,recurrent_grid_properties,final_history,recurrent_history,
     index_protocoll,intensity_protocoll] =  interface_event(
                                         grid_properties,new_grid_properties,new_i_pos,i,
                                         new_intensity,grid_index,new_grid_index,cell_n,
                                         index_protocoll,intensity_protocoll,history,intensity,
                                         printmode,absorp_boundary,absorp_final,mode)


    #First taking care of the terminating rays, passing the focal_plane
   
    #Checking the intensity threshold, even though the rays are terminating, to be consistent
    final_i_pos,final_i_refr,final_intensity,final_history=check_final_intensity(
            final_i_pos,final_i_refr,final_intensity,final_history,intensityborder,smallprintmode)
    
   
    #Few more checks and constructing the simulated picture on the chip
    if history_num<0:
        #Case, where all rays are considered
       
        #Checking, if the rays reach the chip, respecting the aperture of the optics
        #in front of the camera chip (microscope objective)
        checked_final_i_pos_yz,checked_final_intensity=check_aperture(final_i_pos,
                                                                      final_i_refr,
                                                                      aperture,
                                                                      focal_plane_direction,
                                                                      final_intensity,smallprintmode)
       
        #Constructing the image on the chip
        his,yborder,zborder=np.histogram2d(checked_final_i_pos_yz[:,0][:,0],
                                           checked_final_i_pos_yz[:,1][:,0],
                                           bins=chip_resolution,
                                           range=[[focal_plane_size[0],focal_plane_size[1]],
                                                  [focal_plane_size[2],focal_plane_size[3]]],
                                           weights=checked_final_intensity)
       
    else:
        #Case, where only rays with a certain history are considered
        #for example: 1 reflection, 2 reflection, 1 refraction ...
        #history given over history_num
        historychecked_i_pos,historychecked_i_refr,historychecked_intensity=check_final_history(
                final_i_pos,final_i_refr,final_intensity,final_history,smallprintmode,history_num)
       
        #Checking, if the rays reach the chip, respecting the aperture of the optics
        #in front of the camera chip (microscope objective)
        checked_final_i_pos_yz,checked_final_intensity=check_aperture(
                historychecked_i_pos,historychecked_i_refr,aperture,focal_plane_direction,historychecked_intensity,smallprintmode)       
       
        #Constructing the image on the chip
        his,yborder,zborder=np.histogram2d(checked_final_i_pos_yz[:,0][:,0],
                                           checked_final_i_pos_yz[:,1][:,0],
                                           bins=chip_resolution,
                                           range=[[focal_plane_size[0],focal_plane_size[1]],
                                                  [focal_plane_size[2],focal_plane_size[3]]],
                                           weights=checked_final_intensity)

    #Cumulatively adding the pictures of every loop
    final=final+his
   
   
    #Now taking care of the recurrent rays, which stay in the simulation
   
    #Checking the intensity threshold and otherwise deleting rays
    [checked_i_pos,checked_i,checked_intensity,checked_grid_index,checked_grid_properties,
     checked_history]    =  check_intensity(
                            recurrent_i_pos,recurrent_i,recurrent_intensity,recurrent_grid_index,
                            recurrent_grid_properties,recurrent_history,intensityborder,smallprintmode)  
   
    return [checked_i_pos,checked_i,checked_intensity,checked_grid_index,checked_grid_properties,
            final,index_protocoll,intensity_protocoll,checked_history,absorp_change_out]



def costhetai(i,n):
    '''i:Nx3-array incoming beams, n:Nx3-array surface normal
    Returns Nx1-array with cos of the angle between i and n
    Calculates the dot product of the two vectors'''
    return fast_thetai(i[:,0],i[:,1],i[:,2],n[:,0],n[:,1],n[:,2])
@vectorize(['float64(float64,float64,float64,float64,float64,float64)'],target='cpu')
def fast_thetai(ix,iy,iz,nx,ny,nz):
    return ix*nx+iy*ny+iz*nz


def costhetat(i,n,eta1,eta2): 
    '''i:Nx3-array incoming beams, n:Nx3-array surface normal, eta1&eta2:Nx1-arrays refractive indices
    Returns Nx1-array with cos of the angle between n and the transmitted beam
    eta1 corresponds to the material of the incoming beams'''
    return fast_thetat(i[:,0],i[:,1],i[:,2],n[:,0],n[:,1],n[:,2],eta1,eta2)
@vectorize(['float64(float64,float64,float64,float64,float64,float64,float64,float64)'],target='cpu')
def fast_thetat(ix,iy,iz,nx,ny,nz,eta1,eta2):
    return math.sqrt(1-(eta1/eta2)**2*(1-(ix*nx+iy*ny+iz*nz)**2))


def Re(i,n,eta1,eta2):
    '''i:Nx3-array incoming beams, n:Nx3-array surface normal, eta1&eta2:Nx1-arrays refractive indices
    Return Nx1-array with reflectivity
    eta1 corresponds to the material of the incoming beams'''
    a=fast_thetai(i[:,0],i[:,1],i[:,2],n[:,0],n[:,1],n[:,2])
    b=fast_thetat(i[:,0],i[:,1],i[:,2],n[:,0],n[:,1],n[:,2],eta1,eta2)
    #vertpar() returns the mean reflectivity for vertical and parallel polarization
    return vertpar(a,b,eta1,eta2)
@vectorize(['float64(float64,float64,float64,float64)'],target='cpu')
def vertpar(a,b,eta1,eta2):
    return 0.5*(((eta1*a-eta2*b)/(eta1*a+eta2*b))**2+((eta2*a-eta1*b)/(eta2*a+eta1*b))**2)


def checkTIR(i,n,eta1,eta2):
    '''i:Nx3-array incoming beams, n:Nx3-array surface normal, eta1&eta2:Nx1-arrays refractive indices
    Returns Nx1-boolean-array, checking if total internal reflection is taking place
    eta1 corresponds to the material of the incoming beams
    True means TIR is happening'''
    return fckTir(i[:,0],i[:,1],i[:,2],n[:,0],n[:,1],n[:,2],eta1,eta2)
@vectorize(['boolean(float64,float64,float64,float64,float64,float64,float64,float64)'],target='cpu')
def fckTir(ix,iy,iz,nx,ny,nz,eta1,eta2):
    return 1-fast_thetai(ix,iy,iz,nx,ny,nz)**2>(eta2/eta1)**2


def normalize(i):
    '''i:Nx3-array
    Returns i normalized to length 1'''
    i_normalized=np.zeros(np.shape(i)).astype(np.float64)
    if len(i)>0:
        a=np.zeros(np.shape(i)[0]).astype(np.float64)
        gu_fn(i,a,i_normalized)
    return i_normalized
def fn(i,a,i_normalized):
    m, g = i.shape
    for k in range(m):
        for j in range(g):
            a[k] += i[k,j]**2
        for j in range(g):
            i_normalized[k,j]=i[k,j]/math.sqrt(a[k])       
gu_fn = numba.guvectorize(['float64[:,:], float64[:], float64[:,:]'],
                              '(m,n),(m)->(m,n)',target='cpu',nopython=True)(fn)



def reflected_direction(i,n):
    '''i:Nx3-array incoming beams, n:Nx3-array surface normal
    Returns Nx3-array with the reflection direction'''
    reflected_i= np.zeros(i.shape).astype(np.float64)
    if len(i)>0:
        thet=np.float64(0)
        gu_refldir(i,n,thet,reflected_i)
    return reflected_i
def refldir(i,n,thet,reflected_i):
    m, g = i.shape
    for k in range(m):
        thet=fast_thetai(i[k,0],i[k,1],i[k,2],n[k,0],n[k,1],n[k,2])
        for j in range(g):
            reflected_i[k,j]=i[k,j]-2*thet*n[k,j]
   
gu_refldir = numba.guvectorize(['float64[:,:], float64[:,:],float64, float64[:,:]'],
                              '(m,n),(m,n),()->(m,n)',target='cpu',nopython=True)(refldir)



def refracted_direction(i,n,eta1,eta2):
    '''i:Nx3-array incoming beams, n:Nx3-array surface normal, eta1&eta2:Nx1-arrays refractive indices
    Returns Nx3-array with the refraction direction
    eta1 corresponds to the material of the incoming beams'''
    i_refracted= np.zeros(i.shape).astype(np.float64)
    if len(i)>0:
        #commented: slow alternative
        #m, g = i.shape
        #for k in range(m):
        #    a=eta1[k]/eta2[k]
        #    i_refracted[k,:]=a*np.cross(n[k,:],np.cross(-n[k,:],i[k,:]))-n[k,:]*np.sqrt(1-(a**2)*np.dot(np.cross(n[k,:],i[k,:]),np.cross(n[k,:],i[k,:])))
        gu_refrdir(i,-n,eta1,eta2,i_refracted)
    return i_refracted
def refrdir(i,n,eta1,eta2,i_refracted):
    m, g = i.shape
    for k in range(m):
        a=eta1[k]/eta2[k]
        b=(n[k,1]*i[k,2]-n[k,2]*i[k,1])**2+(n[k,2]*i[k,0]-n[k,0]*i[k,2])**2+(n[k,0]*i[k,1]-n[k,1]*i[k,0])**2
        i_refracted[k,0]=a*(-n[k,1]*n[k,0]*i[k,1]+n[k,1]*n[k,1]*i[k,0]+n[k,2]*n[k,2]*i[k,0]-n[k,2]*n[k,0]*i[k,2])-n[k,0]*math.sqrt(1-(a**2)*b)  
        i_refracted[k,1]=a*(-n[k,2]*n[k,1]*i[k,2]+n[k,2]*n[k,2]*i[k,1]+n[k,0]*n[k,0]*i[k,1]-n[k,0]*n[k,1]*i[k,0])-n[k,1]*math.sqrt(1-(a**2)*b)  
        i_refracted[k,2]=a*(-n[k,0]*n[k,2]*i[k,0]+n[k,0]*n[k,0]*i[k,2]+n[k,1]*n[k,1]*i[k,2]-n[k,1]*n[k,2]*i[k,1])-n[k,2]*math.sqrt(1-(a**2)*b)  
        
gu_refrdir = numba.guvectorize(['float64[:,:], float64[:,:],float64[:],float64[:], float64[:,:]'],
                              '(m,n),(m,n),(m),(m)->(m,n)',target='cpu',nopython=True)(refrdir)




def check_aperture(i_pos,i,apert,focal_plane_direction,intensity,smallprintmode):
    """Reduce the number of rays to only those rays fulfilling the numerical aperture condition"""
    good=np.zeros(len(i),dtype=bool)
    if len(i)>0:
        gu_fckapert(i,focal_plane_direction,apert,good)
    ind=np.argwhere(focal_plane_direction==0)
    if smallprintmode:
        #if np.sum((good-1)*(-1))>0:
        print "number of aperturefails " +str(np.sum((good-1)*(-1)))           
    return i_pos[good][:,ind],intensity[good]
def fckapert(i,fpdir,apert,t):
    for k in range(len(i)):
        t[k]=apert**2>1-fast_thetai(i[k,0],i[k,1],i[k,2],fpdir[0],fpdir[1],fpdir[2])**2

gu_fckapert = numba.guvectorize(['float64[:,:], float64[:],float64,boolean[:]'],
                              '(m,n),(n),()->(m)',target='cpu',nopython=True)(fckapert)


def gen_sur(i):
    '''i:Nx3-array incoming beams
    Returns 3x Nx3-arrays cartesian normal surface vectors that can be hit
    considering the sign of the direction vector of the light i'''  
    n1=np.zeros(np.shape(i))
    n2=np.zeros(np.shape(i))
    n3=np.zeros(np.shape(i))
       
    n1[:,0]=np.sign(i[:,0])
    n2[:,1]=np.sign(i[:,1])
    n3[:,2]=np.sign(i[:,2])
    return n1,n2,n3



def aim(i,i_pos,n_pos1,n_pos2,n_pos3):
    '''i:Nx3-array incoming beams, i_pos:Nx3-array beam origins,
    n_pos1&n_pos2&n_pos3:Nx2-arrays x-,y- and z-borders of the box,
    Returns Nx3-array of normal vectors of hit surfaces and
    Nx1-array distances from i_pos to hit surfaces in units of i'''   
    #lamb will carry the information how much time the rays take to reach each
    #cell interface, that could be hit considering the signs of the xyz-components
    #of the direction of the rays.
    lamb=np.zeros(np.shape(i)).astype(np.float64)
    if len(i)>0:       
        gu_aim_lamb(i,i_pos,n_pos1,n_pos2,n_pos3,lamb)   
    h=np.arange(len(i))
    n_label=np.argmin(lamb,axis=1)
    time_distance=np.min(lamb,axis=1)
    n1,n2,n3=gen_sur(i)   
    all_n=np.array([n1,n2,n3])
    cell_n=all_n[n_label,h]
    return cell_n,time_distance
def aim_lamb(i,i_pos,n_pos1,n_pos2,n_pos3,lamb):
    for k in range(len(i)):
        if i[k,0]>0:
            lamb[k,0]=(n_pos1[k,1]-i_pos[k,0])/i[k,0]
        elif i[k,0]<0:
            lamb[k,0]=(n_pos1[k,0]-i_pos[k,0])/i[k,0]
        else:
            lamb[k,0]=np.inf
            
        if i[k,1]>0:
            lamb[k,1]=(n_pos2[k,1]-i_pos[k,1])/i[k,1]
        elif i[k,1]<0:
            lamb[k,1]=(n_pos2[k,0]-i_pos[k,1])/i[k,1]       
        else:
            lamb[k,1]=np.inf

        if i[k,2]>0:
            lamb[k,2]=(n_pos3[k,1]-i_pos[k,2])/i[k,2]
        elif i[k,2]<0:
            lamb[k,2]=(n_pos3[k,0]-i_pos[k,2])/i[k,2]
        else:
            lamb[k,2]=np.inf
gu_aim_lamb = numba.guvectorize(['float64[:,:], float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:]'],
                              '(m,n),(m,n),(m,p),(m,p),(m,p)->(m,n)',target='cpu',nopython=True)(aim_lamb)

def fast_part_prop(i,i_pos,time_distance,grid_properties7,intensity):
    m,n=np.shape(i_pos)
    inpu=np.zeros([m,n+2]).astype(np.float64)
    final=np.zeros([m,n+2]).astype(np.float64)
    inpu[:,:3]=i_pos
    inpu[:,3]=time_distance
    #multiplication of time_distance with i to transorm the time_distance to a
    #spatial distance not necessary, because the length of i is noramlized to 1.
    inpu[:,4]=grid_properties7
    gu_part_prop(i,inpu,intensity,final)
    return final[:,:3],final[:,3],final[:,4]
def part_prop(i,inpu,intensity,final):
    for k in range(len(i)):
        final[k,0]=inpu[k,0]+inpu[k,3]*i[k,0]
        final[k,1]=inpu[k,1]+inpu[k,3]*i[k,1]
        final[k,2]=inpu[k,2]+inpu[k,3]*i[k,2]
        final[k,3]=intensity[k]*math.exp(-inpu[k,4]*inpu[k,3])
        final[k,4]=intensity[k]-final[k,3]

gu_part_prop = numba.guvectorize(['float64[:,:],float64[:,:],float64[:], float64[:,:]'],
                              '(m,n),(m,p),(m)->(m,p)',target='cpu',nopython=True)(part_prop)  
    

def interface_event(grid_properties,new_grid_properties,new_i_pos,i,new_intensity,
                    grid_index,new_grid_index,cell_n,index_protocoll,intensity_protocoll,
                    history,intensity,printmode,absorp_boundary,absorp_final,mode):
   
    #Saving the grid_index or whatever else is needed to trace rays in testmode
    if mode=='test':
        index_protocoll.append(np.copy(grid_index))
    else:
        index_protocoll.append(0)

    #saving intensities for simulation control   
    gesin=np.sum(intensity)
    absin=gesin-np.sum(new_intensity)   
       
    same,mirror,tir,refl,out,refr,finale=interface_decision(i,cell_n,grid_properties[:,6],
                                        new_grid_properties[:,6:],absorp_boundary,absorp_final)
   
    #after interface_decision() has determined which rays undergo which processes
    #interface_event() has to create new arrays to store more lightrays, since the
    #occuring of reflection and refraction splits one ray into two. Therefore the
    #total number of rays increases. All reflected rays and rays, which do not split,
    #overwrite the old rays. Rays that get refracted are saved in one of two new arrays.
    #Either they are written in the array containing the rays, which pass the
    #(terminating) focal plane, or they are written in an array for rays, that
    #stay in the simulation. The array of rays staying in the simulation is then
    #concatenated to the array of reflected and not-splitting rays.
   
    reflectivity=np.zeros(len(i))
    reflectivity[refl]=Re(i[refl],cell_n[refl],grid_properties[:,6][refl],new_grid_properties[:,6][refl])
    #
    sidin=np.sum(new_intensity[out]*(1-reflectivity[out]))
    #
    extra_intensity=new_intensity[refr]*(1.-reflectivity[refr])
    final_intensity=new_intensity[finale]*(1.-reflectivity[finale])
    new_intensity[refl]=new_intensity[refl]*reflectivity[refl]
    #
    extra_i=refracted_direction(i[refr],cell_n[refr],grid_properties[:,6][refr],new_grid_properties[:,6][refr])
    final_i=refracted_direction(i[finale],cell_n[finale],grid_properties[:,6][finale],new_grid_properties[:,6][finale])
    i[mirror+refl+tir]=reflected_direction(i[mirror+refl+tir],cell_n[mirror+refl+tir])
    #
    grid_properties[same]=new_grid_properties[same]
    grid_index[same]=new_grid_index[same]
    extra_grid_properties=new_grid_properties[refr]
    extra_grid_index=new_grid_index[refr]
    #
    final_i_pos=new_i_pos[finale]
    recurrent_i_pos=np.concatenate((new_i_pos,new_i_pos[refr]),axis=0)
    recurrent_i=np.concatenate((i,extra_i),axis=0)
    recurrent_intensity=np.concatenate((new_intensity,extra_intensity),axis=0)
    recurrent_grid_index=np.concatenate((grid_index,extra_grid_index),axis=0)
    recurrent_grid_properties=np.concatenate((grid_properties,extra_grid_properties),axis=0)
    #
    history[mirror]=history[mirror]+1
    history[tir]=history[tir]+10**6#0.001
    extra_history=history[refr]+0.001#10**6
    final_history=history[finale]
    history[refl]=history[refl]+1000
    recurrent_history=np.concatenate((history,extra_history),axis=0)
   
    #Printer and intensity information saver
    samin=np.sum(new_intensity[same])
    mirin=np.sum(new_intensity[mirror])
    tirin=np.sum(new_intensity[tir])
    #sidin=np.sum(new_intensity[out])#*(1-reflectivity[out]))
    refrin=np.sum(extra_intensity)
    reflin=np.sum(new_intensity[refl])
    finin=np.sum(final_intensity)
    #check=absin+samin+mirin+tirin+sidin+refrin+reflin+finin   
   
    testi=[gesin,absin,samin,mirin,tirin,sidin,refrin,reflin,finin,len(intensity)]       
    intensity_protocoll.append(testi)
      
    if printmode:
        print "Incoming "+str(len(intensity))+" with intensity "+str(gesin)[:9]+ " as 100%"
        print "Absorption loss "+str(absin)[:9]+" intensity making "+str(absin/gesin*100.)[:5]+"%"
        print 'Same '+str(np.sum(same))+" with intensity "+str(samin)[:9]+ " making "+ str(samin/gesin*100.)[:5]+"%"
        print 'Mirrored '+str(np.sum(mirror))+" with intensity "+str(mirin)[:9]+ " making "+ str(mirin/gesin*100.)[:5]+"%"   
        print 'TIR '+str(np.sum(tir))+" with intensity "+str(tirin)[:9]+ " making "+ str(tirin/gesin*100.)[:5]+"%"
        print 'Sides '+str(np.sum(out))+" with intensity "+str(sidin)[:9]+ " making "+ str(sidin/gesin*100.)[:5]+"%"   
        print 'Refracted '+str(np.sum(refr))+" with intensity "+str(refrin)[:9]+ " making "+ str(refrin/gesin*100.)[:5]+"%"
        print 'Reflected '+str(np.sum(refl))+" with intensity "+str(reflin)[:9]+ " making "+ str(reflin/gesin*100.)[:5]+"%"
        print 'Finish '+str(len(final_intensity))+" with intensity "+str(finin)[:9]+ " making "+ str(finin/gesin*100.)[:5]+"%"
        print "" 
   
    return [final_i_pos,final_i,final_intensity, recurrent_i_pos,recurrent_i,
        recurrent_intensity,recurrent_grid_index,recurrent_grid_properties,
        final_history,recurrent_history,index_protocoll,intensity_protocoll]   

def interface_decision(i,cell_n,grid_properties_6,new_grid_properties_67,absorp_boundary,absorp_final):
    inpu=np.zeros([len(i),7])
    out=np.zeros([len(i),7],dtype=bool)
    inpu[:,:3]=i
    inpu[:,3:6]=cell_n
    inpu[:,6]=grid_properties_6
    if len(i)>0:
        gu_interface_decision(inpu,new_grid_properties_67,absorp_boundary,absorp_final,out)    
    return out[:,0],out[:,1],out[:,2],out[:,3],out[:,4],out[:,5],out[:,6]
def g_interface_decision(inpu,newgrid,absorp_boundary,absorp_final,out):
    for k in range(len(inpu)):
        #k iterates over the rays
        if inpu[k,6]==newgrid[k,0]:
            #same refractive index on both sides of the interface
            if newgrid[k,1]==absorp_final:
                #new cell is part of the terminating, focal plane
                out[k,6]=True
            else:
                #same refractive index, ray stays in simulation
                out[k,0]=True
        elif newgrid[k,0]==0:
            #refractive index is 0 -> 100% reflectivity
            out[k,1]=True
        else:
            if fckTir(inpu[k,0],inpu[k,1],inpu[k,2],inpu[k,3],inpu[k,4],inpu[k,5],inpu[k,6],newgrid[k,0]):
                #Total Internal Reflection
                out[k,2]=True
            else:
                out[k,3]=True
                if newgrid[k,1]==absorp_boundary:
                    #new cell is part of the spatial boundary of the simulation
                    out[k,4]=True
                else:
                    if newgrid[k,1]==absorp_final:
                        #new cell is part of the terminating, focal plane
                        out[k,6]=True
                    else:
                        #normal refraction
                        out[k,5]=True
gu_interface_decision = numba.guvectorize(['float64[:,:], float64[:,:],float64,float64,boolean[:,:]'],
                              '(m,n),(m,p),(),()->(m,n)',target='cpu',nopython=True)(g_interface_decision)


   

def construct_grid_index(i_pos,grid):
    '''i_pos:Nx3-array beam origins,
    grid:XxYxZx8-array xyz-grid in box-borders with refr.ind. and abs.coeff.
    Returns Nx3-array with x-box-index, y-box-index and z-box-index'''
    x_bins=grid[:,1,1][:,0].tolist()
    y_bins=grid[1,:,1][:,2].tolist()
    z_bins=grid[1,1,:][:,4].tolist()
   
    x_bins.append(np.inf)
    y_bins.append(np.inf)
    z_bins.append(np.inf)   
   
    grid_index=np.ndarray(np.shape(i_pos))

    grid_index[:,0]=np.digitize(i_pos[:,0],x_bins)-1
    grid_index[:,1]=np.digitize(i_pos[:,1],y_bins)-1
    grid_index[:,2]=np.digitize(i_pos[:,2],z_bins)-1
    return grid_index


def construct_properties_from_index(grid_index,grid):
    '''grid_index:Nx3-array box-index,
    grid:XxYxZx8-array xyz-grid in box-borders with refr.ind. and abs.coeff.
    Returns Nx8-array which has full box information
    (xyz-borders, refr.ind. and abs.coeff.) pulled from grid'''
    grid_properties=grid[grid_index[:,0].astype(dtype=np.int64),
                       grid_index[:,1].astype(dtype=np.int64),
                       grid_index[:,2].astype(dtype=np.int64)]
    return grid_properties


def check_final_intensity(final_i_pos,final_i_refr,final_intensity,final_history,intensityborder,smallprintmode):
    """Use the intensity threshold to reduce the number of rays"""
    bigger=final_intensity>intensityborder
    if smallprintmode:
        print "check_finalintens "+ str(np.sum(bigger))+" of "+str(len(bigger))+" passed"
    return final_i_pos[bigger],final_i_refr[bigger],final_intensity[bigger],final_history[bigger]  


   
def check_intensity(recurrent_i_pos,recurrent_i,recurrent_intensity,recurrent_grid_index,
                    recurrent_grid_properties,recurrent_history,intensityborder,smallprintmode):
    """Use the intensity threshold to reduce the number of rays"""
    bigger=recurrent_intensity>=intensityborder
    if smallprintmode:   
        print "check_retintens "+ str(np.sum(bigger))+" of "+str(len(bigger))+" passed"
   
    return [recurrent_i_pos[bigger],recurrent_i[bigger],recurrent_intensity[bigger],
            recurrent_grid_index[bigger],recurrent_grid_properties[bigger],recurrent_history[bigger]]

       

def check_final_history(final_i_pos,final_i_refr,final_intensity,final_history,smallprintmode,history_num):
    """Reduce the number of rays to only those rays, with the right history
    of only zero, one or two or more reflections or refractions or ..."""
    right_refl=final_history<history_num
    if smallprintmode:
        print "check_finalhistory "+ str(np.sum(right_refl))+" of "+str(len(right_refl))+" passed"
    return final_i_pos[right_refl],final_i_refr[right_refl],final_intensity[right_refl]


def inside_absorp(grid,old_grid_index,absorp_change_out,absorp_cross_section_yz):#assumes homogeneity along the x-axis
    """Maps the absorption loss to the grid cells (except the boundary cells)"""
    xb=np.arange(len(grid[:,0,0,0])+1)-0.5
    yb=np.arange(len(grid[0,:,0,0])+1)-0.5
    zb=np.arange(len(grid[0,0,:,0])+1)-0.5   
    absorp_increase=np.histogramdd(old_grid_index,bins=[xb,yb,zb],weights=absorp_change_out)[0]
    return absorp_cross_section_yz+absorp_increase

def irdistr(isou,number_of_rays,boundary_region=True):
    """Takes an grid-like matrix (x,y,z) of a light ray origin probability distribution
    and maps it to the startvolume, does not need to be normalized necessarily"""
    #resolution does not have to correlate with the grid, can be higher
    i_pos_acc=np.zeros([number_of_rays,3])
    isou_norm=isou/np.max(isou)
    isou_x,isou_y,isou_z=np.shape(isou)

    if boundary_region:
        xbins=np.arange(isou_x+1)/float(isou_x-2)-1/float(isou_x-2)   
        ybins=np.arange(isou_y+1)/float(isou_y-2)-1/float(isou_y-2)
        zbins=np.arange(isou_z+1)/float(isou_z-2)-1/float(isou_z-2)
    else:
        xbins=np.arange(isou_x+1)/float(isou_x)   
        ybins=np.arange(isou_y+1)/float(isou_y)
        zbins=np.arange(isou_z+1)/float(isou_z)       
       
    tr=True
    rem=0   
    while tr:
        i_pos=np.random.uniform(0,1,[number_of_rays,3])
        indx=np.digitize(i_pos[:,0],xbins)
        indy=np.digitize(i_pos[:,1],ybins)
        indz=np.digitize(i_pos[:,2],zbins)
        ind=np.stack((indx,indy,indz)).T-1
        prob=np.random.uniform(0,1,number_of_rays)
        ind=np.array(ind)
        acc=isou_norm[ind[:,0],ind[:,1],ind[:,2]]>=prob
        i_pos_acc_help=i_pos[acc]
        if rem+len(i_pos_acc_help)<number_of_rays:
            i_pos_acc[rem:rem+len(i_pos_acc_help)]=i_pos_acc_help
            rem+=len(i_pos_acc_help)
        else:
            i_pos_acc[rem:]=i_pos_acc_help[:number_of_rays-rem]
            tr=False

    return i_pos_acc


def initialize_ray_positions(startdistr,startvolume_borders,number_of_rays):
    if np.sum(startdistr)!=0:
        i_pos=irdistr(startdistr,number_of_rays)
    else:
        i_pos=np.random.uniform(0,1,[number_of_rays,3])
       
    xlow,xhigh,ylow,yhigh,zlow,zhigh=startvolume_borders
   
    i_pos[:,0]=i_pos[:,0]*(xhigh-xlow)+xlow
    i_pos[:,1]=i_pos[:,1]*(yhigh-ylow)+ylow
    i_pos[:,2]=i_pos[:,2]*(zhigh-zlow)+zlow

    return i_pos


def initialize_ray_directions(opening_angle_x,number_of_rays,mode):
    #initialize random directions for in spherical coordinates to be real isotropic
    if len(mode) < 10:
        #normal case
        direc1=np.random.uniform(90-opening_angle_x,90+opening_angle_x,number_of_rays)
        direc2a=np.random.uniform(90-opening_angle_x,90+opening_angle_x,number_of_rays/2)
        direc2b=np.random.uniform(270-opening_angle_x,270+opening_angle_x,number_of_rays/2)
       
    else: 
        if mode == 'two_dim_xz':
            #old better use confinement
            #Two dimensional case in x-z-plane
            #Paper_Reproduction
            #direc1=np.random.uniform(90-opening_angle_x,90,number_of_rays)
            #otherwise
            direc1=np.random.uniform(90-opening_angle_x,90+opening_angle_x,number_of_rays)
            opening_angle_x=opening_angle_x*0.0000001
            direc2a=np.random.uniform(90-opening_angle_x,90+opening_angle_x,number_of_rays/2)
            direc2b=np.random.uniform(270-opening_angle_x,270+opening_angle_x,number_of_rays/2)  
        
        elif mode[:14]== 'confinement_yz':
            #Confined directions
            #Setting y axis to zero reproduces the two_dim_xz case
            opening_angle_y=min(opening_angle_x,int(mode[-5:-3]))
            direc2a=np.random.uniform(90-opening_angle_y,90+opening_angle_y,number_of_rays/2)
            direc2b=np.random.uniform(270-opening_angle_y,270+opening_angle_y,number_of_rays/2)  
            
            opening_angle_z=min(opening_angle_x,int(mode[-2:]))
            direc1=np.random.uniform(90-opening_angle_z,90+opening_angle_z,number_of_rays)
        elif mode[:18]== 'angle_distribution':

            direc2a=np.random.uniform(90-opening_angle_x,90+opening_angle_x,number_of_rays/2)
            direc2b=np.random.uniform(270-opening_angle_x,270+opening_angle_x,number_of_rays/2)
            
            dir_acc=np.zeros(number_of_rays)
            rem=0
            tr=True
            while tr:
                direc=np.random.uniform(90-opening_angle_x,90+opening_angle_x,number_of_rays)
                acc_crit=np.exp(-np.tan((direc-90)/180.*np.pi)**2/(2*np.tan(np.arcsin(np.sin(23*1/(2*np.sqrt(2*np.log(2)))/180.*np.pi)*(1/2.41)))**2))
                prob=np.random.uniform(0,1,number_of_rays)
                acc=prob<=acc_crit
                dir_acc_help=direc[acc]
                
                if rem+len(dir_acc_help)<number_of_rays:
                    dir_acc[rem:rem+len(dir_acc_help)]=dir_acc_help
                    rem+=len(dir_acc_help)
                    #print 'sd'
                else:
                    dir_acc[rem:]=dir_acc_help[:number_of_rays-rem]
                    tr=False
            direc1=dir_acc
        
    diz=np.cos(direc1/180.*np.pi)
    diy1=np.sin(direc1[:number_of_rays/2]/180.*np.pi)*np.cos(direc2a/180.*np.pi)
    diy2=np.sin(direc1[number_of_rays/2:]/180.*np.pi)*np.cos(direc2b/180.*np.pi)
    dix1=np.sin(direc1[:number_of_rays/2]/180.*np.pi)*np.sin(direc2a/180.*np.pi)
    dix2=np.sin(direc1[number_of_rays/2:]/180.*np.pi)*np.sin(direc2b/180.*np.pi)      
           
    dix=np.concatenate((dix1,dix2),axis=0)
    diy=np.concatenate((diy1,diy2),axis=0)
           
    i=np.zeros([number_of_rays,3])
    i[:,0]=dix
    i[:,1]=diy
    i[:,2]=diz

    return i


def mask_volume(i_pos,i,xm1,xm2,ym1,ym2,zm1,zm2):
    """Reduces i_pos and i by all rays in the volume between xm1,xm2 and ym1,ym2 and zm1,zm2"""
    #Construct a list of all the rays in the volume, that shall be masked
    #and then delete these rays from i_pos and i
    masked=[]
    for j in range(len(i_pos[:,0])):
        if i_pos[j,1]>=ym1 and i_pos[j,1]<=ym2:
            if i_pos[j,2]>=zm1 and i_pos[j,2]<=zm2:
                if i_pos[j,0]>=xm1 and i_pos[j,0]<=xm2:
                    masked.append(j)
    i_pos=np.delete(i_pos,masked,0)
    i=np.delete(i,masked,0)
    return i_pos,i


def loop_printer(smallprintmode,printmode,verbleib,counter):
    if smallprintmode:
        if counter%1==0:
            print "Rays still in crystal " +str(verbleib)
            print "Loop ---------------------------------------"+str(counter)
            print ""
    elif printmode:
        print "Rays still in crystal " +str(verbleib)
        print "Loop ---------------------------------------"+str(counter)
        print ""