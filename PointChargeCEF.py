##################### Created by Zhiling Dun on Oct.26 2020 ##############
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.special import wofz
from scipy import integrate
import scipy.linalg as LA
from typing import NamedTuple
import cma
import copy
import sys
from datetime import datetime
from itertools import product


####################################################################################################   
####################################      Setting up         #######################################
####   1=Ce3+, 2=Pr3+, 3=Nd3+, 4=Pm3+, 5=Sm3+, 6=Tb3+, 7=Dy3+, 8=Ho3+, 9=Er3+, 10=Tm3+, 11=Yb3+,####
#################################################################################################### 
RE= 'Ce3+', 'Pr3+', 'Nd3+', 'Pm3+', 'Sm3+', 'Tb3+', 'Dy3+', 'Ho3+', 'Er3+', 'Tm3+', 'Yb3+'
J_total=[2.5, 3.5, 4.5, 4, 2.5, 6, 15/2, 8, 15/2, 6, 7/2]
g_J=[6/7,4/5,8/11,5/3, 2/7, 3/2, 4/3, 5/4, 6/5, 7/6, 8/7]
N_A=6.022e23
emu=1.0783e20 # to muB
muB=0.0578838 # in the unit of  meV/Tesla
meV=11.602  # meV to K
Oe=1e-4 # to Tesla
cm_inverse= 0.12398 # to meV

#intergral number of aplha, beta, gamma
Theta_k=np.array([[-2./35,  2/7./45,  0],  #for Ce3+   
                  [-52./9/25/11,  -4./55/33/3,         17*16./7/121/13/5/81],  #for Pr3+ 
                  [-7./9/121,     -8*17./11/11/13/297, -5.*17*19/27/7/1331/169], #for Nd3+ 
                  [ 14./11/11/15, 952./13/27/1331/5,  2584./121/169/3/63],  #for Pm3+ 
                  [13./7/45,       26./27/35/11,         0    ],            #for Sm3+ 
                  [-1./99,          2./11/1485,        -1./13/33/2079   ],  #for Tb3+ 
                  [-2./9/35,       -8./11/45/273,       4./121/169/27/7  ],  #for Dy3+ 
                  [-1./2/9/25,     -1./2/3/5/7/11/13,  -5./27/7/121/169],    #for Ho3+
                  [4./45/35,       2./11/15/273,       8./169/121/27/7],   #for Er3+
                  [1./99,           8./81/5/121,       -5./13/33/2079],    #for Tm3+
                  [2./63,          -2./77/15,           4./13/33/63]])     #for Yb3+ 
# http://www.mcphase.de/manual/node122.html
lambda_kq=np.array([[1/2, np.sqrt(6), np.sqrt(6)/2, 0, 0, 0, 0],
                    [1/8, np.sqrt(5)/2, np.sqrt(10)/4, np.sqrt(35)/2, np.sqrt(70)/8, 0, 0],
                   [1/16, np.sqrt(42)/8, np.sqrt(105)/16, np.sqrt(105)/8, 3*np.sqrt(14)/16, 3*np.sqrt(77)/8, np.sqrt(231)/16]])  

def Gaussian(x, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha * np.exp(-(x / alpha)**2 * np.log(2))

def Lorentzian(x, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / (x**2 + gamma**2)

#The Voigt line profile, see https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
def Voigt(x, alpha_G, gamma_L):
    """Return the Voigt line shape at x with Lorentzian component HWHM gamma and Gaussian component HWHM alpha."""
    sigma = alpha_G / np.sqrt(2 * np.log(2)) + 1e-9

    return np.real(wofz((x + 1j*gamma_L)/sigma/np.sqrt(2))) / sigma/ np.sqrt(2*np.pi)

#resolution of SEQUOIA high-resolution mode as a function of Ei
def Instrument_resolution(x, Ei):
    a=np.exp(-4.00927)*np.power(Ei,1.02359936)
    b= -0.001585*Ei   -1.06087825
    return np.multiply(a, np.exp(b*x/Ei))

def Findindex(target, Pool):
    for i in range(len(Pool)):
        if target in Pool[i]: 
            return i
    if i == len(Pool): 
        return -1

####################################################################################################   
###################################        Operator Class            #############################
#################################################################################################### 

class Operator():    # total angular momentum operator
    def __init__(self, Jtotal):
        self.J = Jtotal
        self.matrix = np.zeros((int(2*Jtotal+1), int(2*Jtotal+1)))

    def __add__(self,other):
        newobj = Operator(self.J)
        if isinstance(other, Operator):
           newobj.matrix = self.matrix + other.matrix
        else:
           newobj.matrix = self.matrix + other*np.identity(int(2*self.J+1))
        return newobj

    def __radd__(self,other):
        newobj = Operator(self.J)
        if isinstance(other, Operator):
            newobj.matrix = self.matrix + other.matrix
        else:
            newobj.matrix = self.matrix + other*np.identity(int(2*self.J+1))
        return newobj

    def __sub__(self,other):
        newobj = Operator(self.J)
        if isinstance(other, Operator):
            newobj.matrix = self.matrix - other.matrix
        else:
            newobj.matrix = self.matrix - other*np.identity(int(2*self.J+1))
        return newobj

    def __mul__(self,other):
        newobj = Operator(self.J)
        if (isinstance(other, int) or isinstance(other, float) or isinstance(other, complex)):
           newobj.matrix = other * self.matrix
        else:
           newobj.matrix = np.dot(self.matrix, other.matrix)
        return newobj

    def __rmul__(self,other):
        newobj = Operator(self.J)
        if (isinstance(other, int) or isinstance(other, float)  or isinstance(other, complex)):
           newobj.matrix = other * self.matrix
        else:
           newobj.matrix = np.dot(other.matrix, self.matrix)
        return newobj

    def __pow__(self, power):
        newobj = Operator(self.J)
        newobj.matrix = self.matrix
        for i in range(power-1):
            newobj.matrix = np.dot(newobj.matrix,self.matrix)
        return newobj

    def __neg__(self):
        newobj = Operator(self.J)
        newobj.matrix = -self.matrix
        return newobj
    
    def __repr__(self):
        return repr(self.matrix)   
    
####################################################################################################   
###################################       crystal field class        ###############################
####################################################################################################  
INSdata = NamedTuple('INSdata',[('x',np.ndarray),('y',np.ndarray),('yError',np.ndarray),\
                                ('Temperature',float),('Ei',float),('SpecialFWHM',np.ndarray), ('Field',np.ndarray)])
SUSdata = NamedTuple('Susdata',[('x',np.ndarray),('y',np.ndarray),('B',float),('Field',np.ndarray)])
MHdata = NamedTuple('Susdata',[('x',np.ndarray),('y',np.ndarray),('Temperature',float),('Field',np.ndarray)])

class CEFmodel():    
    def __init__(self, Ion_name):
        for i in range(11):
            if (RE[i] == Ion_name): break
            if(i==10):
                print('Initialation failed: wrong rare earth name');return
        self.Ion=i+1
        self.Element=RE[i]      # Element type, e.g. Er3+
        self.Jtotal=J_total[i]  # totol angular momentum quantum number
        self.gJ=g_J[i]          # Lander g-factor
        
        ######       Angular momentum Operator            ##### 
        self.Jz=Operator(self.Jtotal) 
        self.Jplus=Operator(self.Jtotal)  
        self.Jminus=Operator(self.Jtotal)
        self.Jsq=Operator(self.Jtotal)
        for i in range(int(2*self.Jtotal+1)):
            for k in range(int(2*self.Jtotal+1)):
                if i == k:
                    self.Jsq.matrix[i,k]= self.Jtotal*(self.Jtotal+1.)
                    self.Jz.matrix[i,k] = k-self.Jtotal  
                elif k+1 == i:
                    self.Jplus.matrix[i,k] = np.sqrt(self.Jtotal*(self.Jtotal+1)-(k-self.Jtotal)*(k-self.Jtotal+1)) 
                elif k-1 == i:
                    self.Jminus.matrix[i,k] = np.sqrt(self.Jtotal*(self.Jtotal+1)-(k-self.Jtotal)*(k-self.Jtotal-1))                 
        self.Jx=( self.Jplus + self.Jminus)*0.5
        self.Jy=(-self.Jplus + self.Jminus)*0.5j
        
        ######      Expermental dat          #####
        self.EXP_INS=[]     # Inlastic neutron dataset, type= NamedTuple('INSdata')
        self.EXP_SUS=[]     # Suscetipibility dataset, type= NamedTuple('SUSdata')
        self.EXP_MH=[]      # Magnetization dataset, type= NamedTuple('MHdata')
        self.levels_obs=[]  # observed CEF levels
        self.FWHM=0.5         # intrinsic broadening of CEF excitations
        
        ######    Steven Operator and eigenstates          #####
        self.Bkq=[]
        self.Bkq_ini=[]
        self.eigenval=[]
        self.eigenvec=[]
        
        ######    Point Charge Model          #####
        self.NPC=0
        self.PC_variable=[]
        self.PC_value_ini=[]
        self.PC_value=[]
        self.PC_model=[]          # PC model in string format
        self.PC_simpre=[]         # PC model in simpre format
        self.PC_SearchResult=[]   # used to store PC_search results

    ###################################       stevens operator Bkq as in simpre.f            ############    
    def StevensOp(self, k,q):  # http://www.mcphase.de/manual/node124.html
        if   [k,q] == [0,0]:
            Okq    = np.zeros((int(2*J+1), int(2*J+1)))
        elif [k,q] == [1,0]:
            Okq    = self.Jz
        elif [k,q] == [1,1]:
            Okq    = 0.5 *(self.Jplus + self.Jminus)
        elif [k,q] == [1,-1]:
            Okq    = -0.5j *(self.Jplus - self.Jminus)
        elif [k,q] == [2,2]:
            Okq    = 0.5 *(self.Jplus**2 + self.Jminus**2)
        elif [k,q] == [2,1]:
            Okq    = 0.25*(self.Jz*(self.Jplus + self.Jminus) + (self.Jplus + self.Jminus)*self.Jz)
        elif [k,q] == [2,0]:
            Okq    = 3*self.Jz**2 - self.Jsq
        elif [k,q] == [2,-1]:
            Okq    = -0.25j*(self.Jz*(self.Jplus - self.Jminus) + (self.Jplus - self.Jminus)*self.Jz)
        elif [k,q] == [2,-2]:
            Okq    = -0.5j *(self.Jplus**2 - self.Jminus**2)
        elif [k,q] == [4,4]:
            Okq    = 0.5 *(self.Jplus**4 + self.Jminus**4)
        elif [k,q] == [4,3]:
            Okq    = 0.25 *((self.Jplus**3 + self.Jminus**3)*self.Jz + self.Jz*(self.Jplus**3 + self.Jminus**3))
        elif [k,q] == [4,2]:
            Okq    = 0.25 *((self.Jplus**2 + self.Jminus**2)*(7*self.Jz**2 -self.Jsq -5) + (7*self.Jz**2 -self.Jsq -5)*(self.Jplus**2 + self.Jminus**2))
        elif [k,q] == [4,1]:
            Okq    = 0.25 *((self.Jplus + self.Jminus)*(7*self.Jz**3 -(3*self.Jsq+1)*self.Jz) + (7*self.Jz**3 -(3*self.Jsq+1)*self.Jz)*(self.Jplus + self.Jminus))
        elif [k,q] == [4,0]:
            Okq    = 35*self.Jz**4 - (30*self.Jsq -25)*self.Jz**2 + 3*self.Jsq**2 - 6*self.Jsq
        elif [k,q] == [4,-4]:
            Okq    = -0.5j *(self.Jplus**4 - self.Jminus**4)
        elif [k,q] == [4,-3]:
            Okq    = -0.25j *((self.Jplus**3 - self.Jminus**3)*self.Jz + self.Jz*(self.Jplus**3 - self.Jminus**3))
        elif [k,q] == [4,-2]:
            Okq    = -0.25j *((self.Jplus**2 - self.Jminus**2)*(7*self.Jz**2 -self.Jsq -5) + (7*self.Jz**2 -self.Jsq -5)*(self.Jplus**2 - self.Jminus**2))
        elif [k,q] == [4,-1]:
            Okq    = -0.25j *((self.Jplus - self.Jminus)*(7*self.Jz**3 -(3*self.Jsq+1)*self.Jz) + (7*self.Jz**3 -(3*self.Jsq+1)*self.Jz)*(self.Jplus - self.Jminus))
        elif [k,q] == [6,6]:
            Okq    = 0.5 *(self.Jplus**6 + self.Jminus**6)
        elif [k,q] == [6,5]:
            Okq    = 0.25*((self.Jplus**5 + self.Jminus**5)*self.Jz + self.Jz*(self.Jplus**5 + self.Jminus**5))
        elif [k,q] == [6,4]:
            Okq    = 0.25*((self.Jplus**4 + self.Jminus**4)*(11*self.Jz**2 -self.Jsq -38) + (11*self.Jz**2 -self.Jsq -38)*(self.Jplus**4 + self.Jminus**4))
        elif [k,q] == [6,3]:
            Okq    = 0.25*((self.Jplus**3 + self.Jminus**3)*(11*self.Jz**3 -(3*self.Jsq+59)*self.Jz) 
                           + (11*self.Jz**3 -(3*self.Jsq+59)*self.Jz)*(self.Jplus**3 + self.Jminus**3))
        elif [k,q] == [6,2]:
            Okq    = 0.25*((self.Jplus**2 + self.Jminus**2)*(33*self.Jz**4 -(18*self.Jsq+123)*self.Jz**2 +self.Jsq**2 +10*self.Jsq +102) 
                           + (33*self.Jz**4 -(18*self.Jsq+123)*self.Jz**2 +self.Jsq**2 +10*self.Jsq +102)*(self.Jplus**2 + self.Jminus**2))
        elif [k,q] == [6,1]:
            Okq    = 0.25*((self.Jplus +self.Jminus)*(33*self.Jz**5 -(30*self.Jsq-15)*self.Jz**3 +(5*self.Jsq**2 -10*self.Jsq +12)*self.Jz) 
                           + (33*self.Jz**5 -(30*self.Jsq-15)*self.Jz**3 +(5*self.Jsq**2 -10*self.Jsq +12)*self.Jz)*(self.Jplus+ self.Jminus))
        elif [k,q] == [6,0]:
            Okq    = 231*self.Jz**6 - (315*self.Jsq-735)*self.Jz**4 + (105*self.Jsq**2 -525*self.Jsq +294)*self.Jz**2 - 5*self.Jsq**3 + 40*self.Jsq**2 - 60*self.Jsq
        elif [k,q] == [6,-6]:
            Okq    = -0.5j *(self.Jplus**6 - self.Jminus**6)
        elif [k,q] == [6,-5]:
            Okq    = -0.25j*((self.Jplus**5 - self.Jminus**5)*self.Jz + self.Jz*(self.Jplus**5 - self.Jminus**5))
        elif [k,q] == [6,-4]:
            Okq    = -0.25j*((self.Jplus**4 - self.Jminus**4)*(11*self.Jz**2 -self.Jsq -38) + (11*self.Jz**2 -self.Jsq -38)*(self.Jplus**4 - self.Jminus**4))
        elif [k,q] == [6,-3]:
            Okq    = -0.25j*((self.Jplus**3 - self.Jminus**3)*(11*self.Jz**3 -(3*self.Jsq+59)*self.Jz) 
                             + (11*self.Jz**3 -(3*self.Jsq+59)*self.Jz)*(self.Jplus**3 - self.Jminus**3))
        elif [k,q] == [6,-2]:
            Okq    = -0.25j*((self.Jplus**2 - self.Jminus**2)*(33*self.Jz**4 -(18*self.Jsq+123)*self.Jz**2 +self.Jsq**2 +10*self.Jsq +102) 
                             + (33*self.Jz**4 -(18*self.Jsq+123)*self.Jz**2 +self.Jsq**2 +10*self.Jsq +102)*(self.Jplus**2 - self.Jminus**2))
        elif [k,q] == [6,-1]:
            Okq    = -0.25j*((self.Jplus - self.Jminus)*(33*self.Jz**5 -(30*self.Jsq-15)*self.Jz**3 +(5*self.Jsq**2 -10*self.Jsq +12)*self.Jz) 
                             + (33*self.Jz**5 -(30*self.Jsq-15)*self.Jz**3 +(5*self.Jsq**2 -10*self.Jsq +12)*self.Jz)*(self.Jplus - self.Jminus))
        else: print("Wrong [k,q] given!")
        return Okq.matrix  
    
    def addINSdata(self, dataset, Temperature = 5, Ei=30, SpecialFWHM=[0,0,0], Field=[0,0,0]):
        self.EXP_INS.append(INSdata(dataset[:,0], dataset[:,1], dataset[:,2], Temperature, Ei, SpecialFWHM, Field))
        
    def addSUSdata(self, dataset, B, Field=[0,0,0]):
        self.EXP_SUS.append(SUSdata(dataset[:,0], dataset[:,1], B, Field))   
        
    def addMHdata(self, dataset, Temperature, Field=[0,0,0]):
        self.EXP_MH.append(MHdata(dataset[:,0], dataset[:,1], Temperature, Field))   
        
    def clearPC(self):
        self.NPC=0
        self.PC_model=[]
        self.PC_simpre=[]
        
    def addPC(self, PCString):
        self.NPC=self.NPC+1
        self.PC_model.append(PCString.split(','))
        #self.PC_model_string.append('  ' + str(self.NPC) + '    ' + PCString)
        
    def simpre(self):   # update PC_simpre, simpre.data, and run run_simpre_sphere
        if os.path.exists("simpre.dat"):
            os.remove("simpre.dat")
        if len(self.PC_value)==0:
            self.PC_value=copy.deepcopy(self.PC_value_ini)        
        string=''
        for n in range(self.NPC):
            string=string + '  ' + str(n+1) + '  '
            value=np.linspace(0,0,4)
            for i, term in enumerate(self.PC_model[n]):
                for j, term2 in enumerate(self.PC_variable):
                    if term2 in term: break            
                if j!= len(self.PC_variable):
                    value[i]=eval(term, {}, {self.PC_variable[j]: self.PC_value[j]})
                else:
                    value[i]=eval(term)
                string=string + "{0:>11.7f}".format(value[i])+ '  '  
            string=string + '\n'
            self.PC_simpre.append(value)
                
        
        with open("simpre.dat", "w+") as f1:
            print('1=Ce3+, 2=Pr3+, 3=Nd3+, 4=Pm3+, 5=Sm3+, 6=Tb3+, 7=Dy3+, 8=Ho3+, 9=Er3+, 10=Tm3+, 11=Yb3+, 12=user' + '\n', file=f1)
            print(' ' + str(self.Ion)+'    !ion code (from 1 to 12, see above)', file=f1)
            print('  ' + str(self.NPC)+'    !number of effective charges', file=f1)
            print(string+'\n\n', file=f1)
        f1.close
        os.system("./run_simpre_sphere")
        #print("Simpre.dat generated successfully, executive run_simpre_sphere, done! \n")
         
                    
    #######################  read Bkq from file #########################################
    
    def readBkq(self, filename='simpre.out', unit='meV', convention='Steven', printcontol='yes'): #read Bkq, convernt unit if Wybourne given
        Bkq=[]     # have three element, k ,q,  Bkq
        with open(filename,'r') as myfile:
            start,end=1e6,1e6
            for i, line in enumerate(myfile,1):
                if str('Bkq') in line:
                    start=i+2
                elif i>start and line.strip().split()==[]:  
                    end=i     
                    break
                elif i>start and i<end and abs(float(line.strip().split()[2]))> 1e-6:
                    Bkq.append(line.strip().split())               
        myfile.close()    

        Bkq = np.asarray(Bkq,dtype='float')   #convert list to float

        if len(Bkq[0])==4:            #4 colown for simpre.f, thrid column is Amn
            Bkq[:,2]=Bkq[:,3]

        for i in range(len(Bkq)):       # convert energy unit     
            if unit == 'cm^-1':            
                Bkq[i][2]=Bkq[i][2]*cm_inverse   
            if unit == 'K':
                Bkq[i][2]=Bkq[i][2]/meV   
            if convention == 'Wybourne':        # convert Wybourne to Steven coeffiecient
                Bkq[i]=WybournetoSteven(Bkq[i],self.Element)    

        self.Bkq = np.array(Bkq[:,:3])  
        self.eigenval = self.eigensys()[0]
        self.eigenvec = self.eigensys()[1]
        
        if printcontol=='yes':
            np.set_printoptions(suppress=True)
            print("Read Bkq successfully for ",self.Element,", Number of Steven Operator =",len(Bkq), ", in the unit of", unit)
            print(self.Bkq)    

        
    #######################  Calculate transition Intensities #########################################
    
    def TransitionIntensity(self, i,ip, WF, Energy): #each transition i to ip has xyz component
        Intx=abs(np.dot(np.conjugate(WF[ip]),np.dot(self.Jx.matrix, WF[i])))**2
        Inty=abs(np.dot(np.conjugate(WF[ip]),np.dot(self.Jy.matrix, WF[i])))**2
        Intz=abs(np.dot(np.conjugate(WF[ip]),np.dot(self.Jz.matrix, WF[i])))**2
        Int=Intx+Inty+Intz
        return Int,Intx,Inty,Intz
    
    #######################  diagnalize CEF Hamiltonian from Bkq #########################################
    
    def eigensys(self, H_ext=[0,0,0]):   #Bkq[] have three element, k, q, Bkq(meV), H_ext(Tesla)
        H = np.zeros((int(2*self.Jtotal+1), int(2*self.Jtotal+1)),dtype = complex)
        H += -self.gJ*muB*(H_ext[0]*self.Jx.matrix + H_ext[1]*self.Jy.matrix + H_ext[2]*self.Jz.matrix)
        for i in range(len(self.Bkq)):
            H += self.Bkq[i][2]* self.StevensOp(round(self.Bkq[i][0]),round(self.Bkq[i][1]))    
        diagonalH = LA.eigh(H)
        eigenvalues = diagonalH[0] - np.amin(diagonalH[0])
        eigenvectors = diagonalH[1].T
        tol = 1e-13
        eigenvalues[abs(eigenvalues) < tol] = 0.0
        eigenvectors[abs(eigenvectors) < tol] = 0.0
        return eigenvalues,eigenvectors,H

        
    #######################  g-factor   ########################################
    def gtensor(self):      #calculate gfactor according to Steven operators 
        Energy,WF,Hamiltonian =self.eigensys()   #Input Wave function, Energy
        Jx,Jy,Jz=self.Jx.matrix, self.Jy.matrix, self.Jz.matrix
        eigenvec1=WF[0]
        eigenvec2=WF[1]
        Sx=[[np.conjugate(eigenvec1).dot(Jx.dot(eigenvec1)),np.conjugate(eigenvec1).dot(Jx.dot(eigenvec2))],
            [np.conjugate(eigenvec2).dot(Jx.dot(eigenvec1)),np.conjugate(eigenvec2).dot(Jx.dot(eigenvec2))]]
        Sy=[[np.conjugate(eigenvec1).dot(Jy.dot(eigenvec1)),np.conjugate(eigenvec1).dot(Jy.dot(eigenvec2))],
            [np.conjugate(eigenvec2).dot(Jy.dot(eigenvec1)),np.conjugate(eigenvec2).dot(Jy.dot(eigenvec2))]] 
        Sz=[[np.conjugate(eigenvec1).dot(Jz.dot(eigenvec1)),np.conjugate(eigenvec1).dot(Jz.dot(eigenvec2))],
            [np.conjugate(eigenvec2).dot(Jz.dot(eigenvec1)),np.conjugate(eigenvec2).dot(Jz.dot(eigenvec2))]]
        g=np.multiply(np.matrix([[Sx[1][0].real, Sx[1][0].imag, Sx[0][0].real],
           [Sy[1][0].real, Sy[1][0].imag, Sy[0][0].real],
           [Sz[1][0].real, Sz[1][0].imag, Sz[0][0].real]]), 2*self.gJ)
        ### Pesduo-spin rotation
        def Ry(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        def Pesduospinrotation(theta):
            A=g.dot(np.linalg.inv(Ry(theta)))
            B=A-A.T
            return B.item((2, 0))
        x=optimize.fsolve(np.vectorize(Pesduospinrotation), 0, xtol=1e-6)   
        g2=np.dot(g,np.linalg.inv(Ry(x[0])))
        return g2
    
    def diagonalG(self,option='print'):     #diagnoalize gtensor 
        def Ry(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        def Pesduospinrotation(theta):
            A=g.dot(np.linalg.inv(Ry(theta)))
            B=A-A.T
            return B.item((2, 0))
        g=self.gtensor()
        x=optimize.fsolve(np.vectorize(Pesduospinrotation), 0, xtol=1e-6)      
        g2=np.dot(g,np.linalg.inv(Ry(x[0])))
        
        ### real-spin rotation
        def Realspinrotation(theta):
            A=Ry(theta).dot(g2.dot(np.linalg.inv(Ry(theta)))) 
            return A.item((0, 1))+A.item((0, 2))
        
        Theta=optimize.fsolve(np.vectorize(Realspinrotation), 0, xtol=1e-6)[0]
        A=np.matrix(Ry(Theta))
        Theta_degree=Theta*180/np.pi
        g_diagonal=Ry(Theta).dot(g2.dot(np.linalg.inv(Ry(Theta))))

        if option=='print':
            print(' Rotational matrix = ', '\n', A,'\n', 'Rotation angle =',  Theta_degree,'\n')
            print('g_diagnal = ','\n', g_diagonal)

        return A,Theta_degree,g_diagonal
    
    
    #######################  Evaluate parttern from Bkq at given Tempearture, Ei, Field #########################################
    def Evaluate_pattern(self, Ei = 160, Temperature = 5, dataset='', Field=[0,0,0], FWHM=0.5, SpecialFWHM=[0,0,0],
                         UsePCini=False, Plotcontrol=False, Chi2Method='linear'): 
        if dataset!='':
            xaxis = self.EXP_INS[dataset].x
            Exp = self.EXP_INS[dataset].y
            BG = self.EXP_INS[dataset].yError
            Ei = self.EXP_INS[dataset].Ei
            FWHM = self.FWHM
            Temperature = self.EXP_INS[dataset].Temperature
            Field = self.EXP_INS[dataset].Field
            SpecialFWHM = self.EXP_INS[dataset].SpecialFWHM       
        else:            #no dataset given, calc 0 to Ei meV, with given FWHM
            xaxis = np.linspace(start=0, stop=Ei, num=300)           
        if UsePCini!=False:
            self.PC_value = copy.deepcopy(self.PC_value_ini)
            self.simpre() 
            self.readBkq(filename="simpre.out",printcontol='no')
        Energy,WF,Hamiltonian=self.eigensys(Field)   #Input Wave function, Energy, Hamiltonian, Bkq
        Calc=np.linspace(0, 0, np.size(xaxis))  #calc spectrum
        for i in range(int(2*self.Jtotal+1)):
            for j in range(int(2*self.Jtotal+1)):
                if i<j and abs(Energy[i]-Energy[j])> 0.02*Ei:        # exclude elastic line
                    Convolution=np.linspace(0, 0, len(xaxis))
                    FWHM_Gaussian= Instrument_resolution(Energy[j]-Energy[i],Ei)      #resolution
                    if  SpecialFWHM[0]<Energy[j]-Energy[i]<SpecialFWHM[1]:
                        FWHM= SpecialFWHM[2]  
                    else: 
                        FWHM=self.FWHM
                    for k in range(np.size(xaxis)):
                        Convolution[k]=Voigt(xaxis[k]-Energy[j]+Energy[i], FWHM_Gaussian, FWHM)
                        #Convolution[k]=gamma/(np.pow(xaxis[k]-Energy[j]+Energy[i],2)+gamma*gamma/4)
                    Calc=Calc+self.TransitionIntensity(i,j,WF,Energy)[0]*np.exp(-Energy[i]*11.602/Temperature)*Convolution                
        #Calc=Calc/max(Calc)
        if dataset!='':
            Calc=Calc/np.sum(Calc)*np.sum(Exp) 
        else: Calc=Calc/np.sum(Calc)
        #######  plot spetrum  ########
        if Plotcontrol!=False:
            Frontsize = 12
            plt.plot(xaxis,Calc,'--r', label='Calc.', alpha=0.8)
            if dataset!='':
                plt.errorbar(self.EXP_INS[dataset].x, self.EXP_INS[dataset].y, yerr=self.EXP_INS[dataset].yError, fmt='go',alpha=0.2, label='exp')
            plt.xlabel('E(meV)',fontsize=Frontsize)
            plt.ylabel('Norm. Intensity (a.u.)',fontsize=Frontsize)
            plt.legend(loc='upper right', frameon=False,fontsize=Frontsize)
            plt.xticks(fontsize=Frontsize)
            plt.yticks(fontsize=Frontsize)
            plt.margins(0.1)  
        #######  calculate Chi_2  ########
        Chi2=0           #default Chi2
        if dataset!='':
            N_point=len(xaxis)
            if Chi2Method == 'linear':
                temp=np.divide(Calc-Exp, BG)
                Chi=1/N_point*np.sqrt(temp.dot(temp))
            else:
                temp=(Calc+BG)/(Exp+BG)
                Chi=1/N_point*abs(Calc-Exp).dot(temp+1/temp-2)*100
            return Chi, xaxis,Calc
        else:        
            return xaxis,Calc
        
    #######################  Evalutate Chi_2 from all INS pattern########################################
    count=0
    
    def Chi2_INS(self, X0=[], Fit_variable='', Chi2Method='linear', TargetChi2energy=1.0, FitMethod='PC' ):
        self.count=self.count+1
        if FitMethod == 'PC':
            for i, term in enumerate(Fit_variable):
                if term == 'FWHM':
                    self.FWHM =X0[-1]
                else:
                    index=Findindex(term, self.PC_variable)
                    self.PC_value[index]=X0[i]
            self.simpre()
            self.readBkq(filename="simpre.out",printcontol=0)
        elif FitMethod == 'StevenOp':  # Steven .mJ fit
            if len(X0)!=0:
                self.Bkq[:,2]=X0[:-1] 
                self.FWHM=X0[-1]
        else:
            print('Wrong FitMethod given!'); return
        Chi2_energy=Chi2=0
        #################   use energy to guide       ########### 
        Energy=self.eigensys()[0]
        for i in range(len(Energy)):              
            if self.levels_obs[i]>0:
                Chi2_energy+=(Energy[i]-self.levels_obs[i])**2/Energy[i]
        for i, term in enumerate(self.EXP_INS):
            Chi2=Chi2+self.Evaluate_pattern(self, dataset=i, Chi2Method=Chi2Method)[0]
        
        print('Function',self.count,'evaluated, Chi2_energy = ', "{0:>8.4f}".format(Chi2_energy), ',  Chi2 = ', "{0:>8.4f}".format(Chi2),
              '                                                                               ', end='\r')  
        # fit to energies first, then total spectra
        return min(100,Chi2)*max(1,Chi2_energy/TargetChi2energy)

    #######################  search parameter space given certain boundary of PC_values #######################################
    def PCsearch(self, boundary, filename='PCsearch.dat'):
        self.count=0
        self.PC_SearchResult=[]
        PC_value_grid=[]
        temp=boundary[0]
        for i in range(1,len(boundary)):
            temp=list(product(temp, boundary[i]))
        for i in temp:
            PC_value_grid.append(str(i).replace('(','').replace(')',''))            
        if os.path.exists(filename):
            os.remove(filename)                 
        with open(filename,'a') as f:
            np.set_printoptions(precision=3,suppress=True,linewidth=100)
            for term in PC_value_grid:
                self.PC_value=[float(x) for x in str(term).split(',')]
                Chi2=self.Chi2_INS()
                print('Evaluating No.%d of %d, Chi2_INS = %f' % (self.count, len(PC_value_grid), Chi2), end=10*'   '+'\r')
                temp="{0:9.3f}".format(Chi2)+',   '+ str(term)
                self.PC_SearchResult.append([float(x) for x in temp.split(',')])
                print(temp, end='\n', file=f)   
        f.close()    
        with open(filename,'r+') as f:    #rewite file after sorting
            Data=np.array(self.PC_SearchResult)           
            self.PC_SearchResult=Data[Data[:,0].argsort()]
            np.set_printoptions(precision=3,suppress=True)     
            print("   Chi^2   ",self.PC_variable, end ='\n', file=f)
            for i in self.PC_SearchResult:
                print("{0:10.3f}".format(i[0]), end ='   ', file=f)      
                print(str(i[1:]).replace('[','').replace(']','').replace('\n',''), end='\n', file=f)       
        f.close()
        return 'Done!'

    #########  Ppint charge fit. ###############
    def PCfit(self, Fit_variable = 'All', SearchMethod ='Nelder-Mead', Chi2Method='linear', Bonds= None, Tolfun=1e-3, TargetChi2energy = 1.0):  
        # e.g. Bonds=[[1.5, 1.4, 1.4, 70, 70, 55, 0.40, 0.20, 0.15], [1.8, 1.7, 1.5, 90, 90, 65, 0.60, 0.40, 0.3]]
        self.count=0       
        Fit_value=[]
        if Fit_variable == 'All':
            Fit_variable = list(self.PC_variable)
            Fit_variable.append('FWHM')       
        for i, term in enumerate(Fit_variable):
            if term == 'FWHM':
                Fit_value.append(self.FWHM)
            else:
                index=Findindex(term, self.PC_variable)
                Fit_value.append(self.PC_value_ini[index])
        if SearchMethod == 'Nelder-Mead':
            return optimize.minimize(self.Chi2_INS, Fit_value, args=(Fit_variable, Chi2Method,TargetChi2energy), method= SearchMethod, tol=Tolfun)
        elif SearchMethod == 'cma':    
            if Bonds == None:
                Bonds =[np.array(Fit_value)*0.8, np.array(Fit_value)*1.2]
            return cma.fmin(self.Chi2_INS, Fit_value, 0.01, args=(Fit_variable, Chi2Method,TargetChi2energy),options={'ftarget':Tolfun, 'bounds':Bonds, 'seed':1, 'popsize':20})        
        else:
            print('Please chose either \'Nelder-Mead\' or \'cma\' SearchMethod')
        
    #########  Steven .mJ fit. ###############
    def StevenOpfit(self, Fit_variable = 'All', SearchMethod ='Nelder-Mead', Chi2Method='linear', Bonds= None, Tolfun=1e-3, TargetChi2energy = 1.0):
        self.count=0
        if len(self.Bkq)==0:
            self.Bkq=copy.deepcopy(self.Bkq_ini)
        Fit_variable =list(self.Bkq[:,2])
        Fit_variable.append(self.FWHM)
        return optimize.minimize(self.Chi2_INS, Fit_variable, args=(Fit_variable, Chi2Method,TargetChi2energy,'StevenOp'), method= SearchMethod, tol=Tolfun)
        
    ####################   Susceptibility & MH  ###################
    def Magnetization(self,Temperature, H_ext, Weiss_field=0.0): #with Wiess field correction 
        def FindM(M):
            Field=H_ext - Weiss_field*M*(N_A/emu*Oe)      # minus sign here
            #print(Field)
            eigenval,eigenvec,Ham=self.eigensys(Field)
            Z = np.sum(np.exp(-eigenval*meV/Temperature))
            [Mx,My,Mz]=[0,0,0]
            for i in range(int(2*self.Jtotal+1)):         #conjugate is important
                Mx += self.gJ*np.exp(-eigenval[i]*meV/Temperature)*np.dot(np.conjugate(eigenvec[i]),np.dot(self.Jx.matrix, eigenvec[i]))/Z
                My += self.gJ*np.exp(-eigenval[i]*meV/Temperature)*np.dot(np.conjugate(eigenvec[i]),np.dot(self.Jy.matrix, eigenvec[i]))/Z
                Mz += self.gJ*np.exp(-eigenval[i]*meV/Temperature)*np.dot(np.conjugate(eigenvec[i]),np.dot(self.Jz.matrix, eigenvec[i]))/Z  
            if np.linalg.norm(M) == 0:
                return [Mx.real, My.real, Mz.real]
            else: 
                return np.linalg.norm([Mx,My,Mz]-M)  
        
        M0=FindM(np.array([0,0,0])) # calcualte Magnetization without Weiss field correction
        if Weiss_field == 0:
            return M0
        else:
            return optimize.minimize(FindM, M0, method= 'Nelder-Mead', tol=1e-6, options={'disp': False}).x
          
    def Susceptibility(self, B, Temperature, Weiss_field=0.0):  # calc suceptibility at certain B and T using dM/dH
        # mu0H in the unit of Telsa
        # default weiss field = 0
        dH=1e-4
        H_minus=[[B-dH/2,0,0],[0,B-dH/2,0],[0,0,B-dH/2]]
        H_plus =[[B+dH/2,0,0],[0,B+dH/2,0],[0,0,B+dH/2]]
        Chi=np.zeros(3)        # chi_x, chi_y, chi_z.
        for i in range(3):
            M1=self.Magnetization(Temperature,H_minus[i],Weiss_field)[i]
            M2=self.Magnetization(Temperature,H_plus[i],Weiss_field)[i]
            Chi[i]=(M2-M1)/dH*N_A/emu*Oe        
        Chi_powder=(Chi[0]+Chi[1]+Chi[2])/3        
        return Chi_powder, Chi[0], Chi[1], Chi[2]     # return chi_powder, Chix, Chiy,Chiz,

    def Powder_Magnetization(self, Temperature=1.8, B_range=[], Weiss_field=0.0, dataset=None, Plotcontrol=True, intergration_step=20): 
        def f(phi,theta, B):
            if B==0:  B+= 1e-13
            Hx,Hy,Hz=[B*np.sin(theta)*np.cos(phi),B*np.sin(theta)*np.sin(phi),B*np.cos(theta)]
            VecH=np.array([Hx,Hy,Hz])
            UnitVecH=VecH/np.sqrt(np.sum(VecH**2))
            M=self.Magnetization(Temperature, VecH, Weiss_field)                   
            return UnitVecH.dot(M)*np.sin(theta)/4/np.pi  #sin(theta)/4/pi due to solid angle interation
        ##### chose  step intergration,Number of point = 4step^2
        def powder_average(B, step):
            Theta=np.linspace(0, np.pi, step)
            Phi=np.linspace(0, 2*np.pi, 2*step)
            f_M=np.zeros((step,2*step))
            for i,theta in enumerate(Theta):
                for j, phi in enumerate(Phi):
                    f_M[i][j]=f(phi,theta,B)
            return integrate.simps(integrate.simps(f_M, Phi), Theta)    
        if dataset==None:
            if len(B_range)!=0:
                x_calc=B_range
        else:
            Temperature=self.EXP_MH[dataset].Temperature
            if len(B_range)!=0: 
                x_calc=B_range
                x_exp=[]; y_exp=[] 
                for i, B in enumerate(self.EXP_MH[dataset].x):
                    if  B_range[0]<= B <=B_range[-1]:
                        x_exp.append(B)
                        y_exp.append(self.EXP_MH[dataset].y[i])
            else:
                x_calc=x_exp=self.EXP_MH[dataset].x
                y_exp=self.EXP_MH[dataset].y 
        MH_powder = np.linspace(0,0,len(x_calc))
        for i, B in enumerate(x_calc):
            MH_powder[i] = powder_average(B, intergration_step)
        if Plotcontrol==True:
            Frontsize = 12
            if dataset!=None:
                plt.plot(x_exp, y_exp, 'go',alpha=0.2, label='exp')
            plt.plot(x_calc,MH_powder,'--r', label=' Calc.', alpha=0.8)
            plt.xlabel('$\mu_0 H$ (T)',fontsize=Frontsize)
            plt.ylabel('MH($\mu_B$)',fontsize=Frontsize)
            plt.legend(loc='upper left', frameon=False,fontsize=Frontsize)
            plt.xticks(fontsize=Frontsize)
            plt.yticks(fontsize=Frontsize)
            plt.margins(0.1)  
        return x_calc, MH_powder

    def Powder_InverseSusceptibility(self, B=0.1, Temperature_range=[], Weiss_field=0.0, dataset=None, Plotcontrol=True):
        if dataset==None:
            if len(Temperature_range)!=0:
                x_calc=Temperature_range
            else:    x_calc = np.linspace(1,300,100)
        else:
            B=self.EXP_SUS[dataset].B
            if len(Temperature_range)!=0: 
                x_calc=Temperature_range
                x_exp=[]; y_exp=[]
                for i, Temperature in enumerate(self.EXP_SUS[dataset].x):
                    if  Temperature_range[0]<= Temperature <=Temperature_range[-1]:
                        x_exp.append(Temperature)
                        y_exp.append(self.EXP_SUS[dataset].y[i])
            else:
                x_calc=x_exp=self.EXP_SUS[dataset].x
                y_exp=self.EXP_SUS[dataset].y 
        Chi_powder = np.linspace(0,0,len(x_calc))
        for i, Temperature in enumerate(x_calc):
            Chi_powder[i] = 1/self.Susceptibility(B,Temperature,Weiss_field)[0]      
        if Plotcontrol==True:
            Frontsize = 12
            if dataset!=None:
                plt.plot(x_exp, y_exp, 'go',alpha=0.2, label='exp')
            plt.plot(x_calc,Chi_powder,'--r', label='$\chi_{powder}^{-1}$ Calc.', alpha=0.8)
            plt.xlabel('$T$ (K)',fontsize=Frontsize)
            plt.ylabel('$\chi^{-1}$(emu/mol/Oe)',fontsize=Frontsize)
            plt.legend(loc='upper left', frameon=False,fontsize=Frontsize)
            plt.xticks(fontsize=Frontsize)
            plt.yticks(fontsize=Frontsize)
            plt.margins(0.1)  
        return x_calc, Chi_powder


   #######################  write result to file ########################################
    def writefile(self, filename, FitMethod = 'PC'):
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename,'a') as f:
            np.set_printoptions(precision=7,suppress=True)
            f.write('***********************   Output file generated by PointChargeCEF   ************************\n\n')
            if FitMethod == 'PC':
                print('************************************************\n****     Point Charge fit for %s ion      ****' % self.Element, file=f)
                print('****     Least square fit to INS Patterns   ****', file=f)
                print('************************************************\n  Chi2_INS = ', self.Chi2_INS(self.PC_value), file=f)
                print('\n\n************************************************\n      Point Charge Parameters   \n************************************************', file=f)
                for i,term in enumerate(self.PC_variable):
                    print("{0:>7}".format(self.PC_variable[i]), '_ini = ', self.PC_value_ini[i], end=',     ',file=f)
                    print("{0:>7}".format(self.PC_variable[i]), '_fit = ', self.PC_value[i], file=f)
                print('\n Fitted FWHM = %.3f meV' % self.FWHM, file=f )
                print('-----------------------------------------------\n', file=f)
            else:    
                print('************************************************\n****    Steven Operator fit for %s ion    ****' % self.Element, file=f)
                print('****     Least square fit to INS Patterns   ****', file=f)
                print('************************************************\n  Chi2_INS = ', self.Chi2_INS(FitMethod = 'StevenOp'), end='\n\n', file=f)
                print(10*'*****'+'\n  Fitted Crystal Field Parameters (meV)  \n'+10*'*****'+'\n',file=f)
                f.write('   k         q          Bkq')
                f.write('\n ----------------------------------------------- \n\n')
                f.write(str(self.Bkq).replace('[',' ').replace(']',' '))
                print('\n\n Fitted FWHM = %.3f meV' % self.FWHM, file=f )
                print('-----------------------------------------------\n', file=f)
                print('\n\n************************************************\n Eigen-functions in the total angular momentum basis', file=f)
                print('************************************************\n', file=f)
                f.write(str(self.eigensys()[1].real).replace('[','  ').replace('\n',' ').replace(']','\n').replace(',','    '))
            np.set_printoptions(precision=3,suppress=True)
            f.write('\n\n************************************************\n           CEF energies (meV)     \n')
            f.write('************************************************\n')
            print('Input energies: ', self.levels_obs, file=f)
            print('Fitted energies: ',str(self.eigensys()[0]).replace('[','').replace(']','').replace('\n',' '), file=f)
            f.write('\n************************************************\n              g-factor tensor            ')
            f.write('\n************************************************\n')
            f.write(str(self.gtensor()).replace('[',' ').replace(']',' '))
            f.write('\n -----------------------------------------------\n')
            f.write('Rotation needed to make g-factor diagnal: ')
            rotation, g_diagnal=self.diagonalG(option='not print')[1:3]
            f.write(str(rotation))
            f.write('\n')
            f.write(str(g_diagnal).replace('[',' ').replace(']',' '))
            f.write('\n\n\n')
            
            if FitMethod == 'PC':
                with open('simpre.out') as datafile:
                    datasimpre = datafile.read()
                datafile.close()
                f.write(datasimpre)
            
            for i in range(len(self.EXP_INS)):
                Chi2, x, y=self.Evaluate_pattern(dataset=i)
                f.write('\n\n\n-----------------------------------------------\n')
                print('Pattern %d, Temperature = %.2f K, Ei = %.1f meV, SpecialFWHM = %s' 
                      %(i, self.EXP_INS[i].Temperature, self.EXP_INS[i].Ei, self.EXP_INS[i].SpecialFWHM), file =f)
                f.write('The calculated pattern with Chi = ')
                f.write(str(Chi2).replace('[','').replace(']',''))
                Pattern=[]
                Pattern.append(x)
                Pattern.append(y)
                Pattern.append(self.EXP_INS[i].y)
                Pattern.append(self.EXP_INS[i].yError)
                f.write('\n\n E(meV)    I_fit     I_exp   I_Error  \n')  
                f.write(str(np.transpose(Pattern)).replace('[',' ').replace(']',''))
                f.write('\n')
        f.close()                
        np.set_printoptions(precision=7,suppress=True)
        
  



 ####################################################################################################   
###################################       Unit convert functions        #############################
#################################################################################################### 
def WybournetoSteven(Bkq_Wybourne, Ion):   #Wybourne operators Bqk converts to stevens operator Bkq 
    Ion = Findindex(Ion, RE)+1
    [k,q,Bkq]=Bkq_Wybourne  
    return [k,q,Bkq*Theta_k[Ion-1,int(k/2-0.5)]*lambda_kq[int(k/2-0.5),int(q)]]  


def BkqtoAkq(Bkq, Ion):   #stevens operator Bkq converted to Akq
    Ion = Findindex(Ion, RE)+1
    Akq=copy.deepcopy(Bkq)
    for i in range(len(Bkq)):
        k= Akq[i,0]
        Akq[i,2]=Bkq[i,2]/Theta_k[Ion-1,int(k/2-0.5)]
    return Akq
        
def AkqtoBkq(Akq, Ion):   #Akq converted to Bkq
    Ion = Findindex(Ion, RE)+1
    Bkq=copy.deepcopy(Akq)
    for i in range(len(Bkq)):
        k= Akq[i,0]
        Bkq[i,2]=Akq[i,2]*Theta_k[Ion-1,int(k/2-0.5)]
    return Bkq    
    

def ConvertCEFparamters(Bkq, Ion_from, Ion_to):   #convert CEF paramerters from one ion to another
    Bkq_new=copy.deepcopy(Bkq)
    Ion_from = Findindex(Ion_from, RE)+1
    Ion_to = Findindex(Ion_to, RE)+1
    for i in range(len(Bkq)):
        k= Bkq[i,0]   
        Bkq_new[i,2]=Bkq[i,2]/Theta_k[Ion_from-1,int(k/2-0.5)]*Theta_k[Ion_to-1,int(k/2-0.5)]
    return np.array(Bkq_new)